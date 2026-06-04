import os
import re
import csv
import glob
import argparse
 
RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
 
# consistent colors per config across all figures
CONFIG_COLORS = {"GPU-FP32": "#1f9e89", "GPU-FP16": "#3b6fb6", "CPU": "#b4530a"}
PRECISION_ORDER = ["FP32", "FP16", "INT8"]
 
 
# ----------------------------------------------------------------------------- 
def _agg(df):
    """Mean + std across repeats for each (model, power_mode, config) cell."""
    keys = ["model", "power_mode", "config", "precision", "device"]
    metrics = ["throughput_ips", "perf_per_watt_ips_w", "dynamic_compute_mw_mean",
               "energy_mj_per_inf", "latency_ms_mean", "latency_ms_p99",
               "tj_peak_c"]
    g = df.groupby(keys)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    return out
 
 
# ----------------------------------------------------------------------------- 
def fig_perf_per_watt(agg, plt, np):
    """F1: grouped bars, perf/watt by config, grouped by power mode, per model."""
    for model in sorted(agg["model"].unique()):
        sub = agg[agg["model"] == model]
        modes = sorted(sub["power_mode"].unique())
        configs = [c for c in ["CPU", "GPU-FP16", "GPU-FP32"] if c in set(sub["config"])]
        x = np.arange(len(modes)); w = 0.8 / max(1, len(configs))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for i, cfg in enumerate(configs):
            means = [sub[(sub.power_mode == m) & (sub.config == cfg)]
                     ["perf_per_watt_ips_w_mean"].mean() for m in modes]
            errs = [sub[(sub.power_mode == m) & (sub.config == cfg)]
                    ["perf_per_watt_ips_w_std"].mean() for m in modes]
            ax.bar(x + i * w, means, w, yerr=errs, capsize=3,
                   label=cfg, color=CONFIG_COLORS.get(cfg))
        ax.set_xticks(x + w * (len(configs) - 1) / 2)
        ax.set_xticklabels(modes)
        ax.set_ylabel("Perf-per-watt (inferences/s/W)")
        ax.set_title(f"Efficiency by datapath and power budget — {model} model")
        ax.legend(title="config")
        ax.grid(axis="y", alpha=0.3)
        _save(fig, plt, f"F1_perf_per_watt_{model}")
 
 
def fig_throughput(agg, plt, np):
    """F5: grouped bars, raw throughput by config x power mode, per model."""
    for model in sorted(agg["model"].unique()):
        sub = agg[agg["model"] == model]
        modes = sorted(sub["power_mode"].unique())
        configs = [c for c in ["CPU", "GPU-FP16", "GPU-FP32"] if c in set(sub["config"])]
        x = np.arange(len(modes)); w = 0.8 / max(1, len(configs))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for i, cfg in enumerate(configs):
            means = [sub[(sub.power_mode == m) & (sub.config == cfg)]
                     ["throughput_ips_mean"].mean() for m in modes]
            errs = [sub[(sub.power_mode == m) & (sub.config == cfg)]
                    ["throughput_ips_std"].mean() for m in modes]
            ax.bar(x + i * w, means, w, yerr=errs, capsize=3,
                   label=cfg, color=CONFIG_COLORS.get(cfg))
        ax.set_xticks(x + w * (len(configs) - 1) / 2)
        ax.set_xticklabels(modes)
        ax.set_ylabel("Throughput (inferences/s)")
        ax.set_title(f"Throughput by datapath and power budget — {model} model")
        ax.legend(title="config"); ax.grid(axis="y", alpha=0.3)
        _save(fig, plt, f"F5_throughput_{model}")
 
 
def fig_tradeoff(agg, acc, plt, np):
    """F2: perf/watt vs top-1 accuracy, one point per (model, precision)."""
    if acc is None:
        print("[analyze] no accuracy.csv — skipping trade-off figure")
        return
    # GPU rows only; join perf/watt (mean over modes) with accuracy by (model, precision)
    gpu = agg[agg["device"] == "GPU"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    markers = {"complex": "o", "light": "s"}
    for model in sorted(gpu["model"].unique()):
        for prec in PRECISION_ORDER:
            ppw = gpu[(gpu.model == model) & (gpu.precision == prec)]["perf_per_watt_ips_w_mean"]
            arow = acc[(acc.model == model) & (acc.precision == prec)]
            if ppw.empty or arow.empty:
                continue
            x_acc = float(arow["top1"].iloc[0]) * 100.0
            y_ppw = float(ppw.mean())
            ax.scatter(x_acc, y_ppw, s=90, marker=markers.get(model, "o"),
                       color=_prec_color(prec), edgecolor="black", zorder=3)
            ax.annotate(f"{model}/{prec}", (x_acc, y_ppw),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Top-1 accuracy (%)  —  the COST")
    ax.set_ylabel("Perf-per-watt (inf/s/W)  —  the BENEFIT")
    ax.set_title("Quantization trade-off: efficiency vs accuracy")
    ax.grid(alpha=0.3)
    _save(fig, plt, "F2_tradeoff_ppw_vs_accuracy")
 
 
def _prec_color(p):
    return {"FP32": "#666666", "FP16": "#3b6fb6", "INT8": "#1f9e89"}.get(p, "#999")
 
 
# ----------------------------------------------------------------------------- 
def parse_timeseries(path):
    """Return (rel_seconds, tj_c, compute_mw) from a ts_*.csv (interleaved rows)."""
    bucket = {}  # t -> {"zones": {...}, "rails": {...}}
    with open(path) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) < 5:
                continue
            t, rail, mw, zone, c = row
            try:
                t = float(t)
            except ValueError:
                continue
            b = bucket.setdefault(t, {"zones": {}, "rails": {}})
            if rail and mw:
                b["rails"][rail] = float(mw)
            if zone and c:
                b["zones"][zone] = float(c)
    ts = sorted(bucket.keys())
    if not ts:
        return [], [], []
    t0 = ts[0]
    rel = [t - t0 for t in ts]
    tj = [max(bucket[t]["zones"].values()) if bucket[t]["zones"] else float("nan") for t in ts]
    pw = [bucket[t]["rails"].get("VDD_CPU_GPU_CV",
          bucket[t]["rails"].get("VDD_IN", float("nan"))) for t in ts]
    return rel, tj, pw
 
 
def _label_from_fname(path):
    m = re.match(r"ts_(.+?)_(MAXN|7W)_(.+?)_r(\d+)\.csv", os.path.basename(path))
    return m.groups() if m else (os.path.basename(path), "", "", "")
 
 
def fig_thermal_timeseries(plt, np, fan_threshold=70.0):
    """F3: per-trial temp & power vs time (one fig per ts file)."""
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "ts_*.csv"))):
        rel, tj, pw = parse_timeseries(path)
        if not rel:
            continue
        model, mode, cfg, rep = _label_from_fname(path)
        fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
        ax1.plot(rel, tj, color="#c0392b", label="junction temp")
        ax1.axhline(fan_threshold, ls="--", color="#c0392b", alpha=0.5,
                    label=f"fan-on threshold {fan_threshold:.0f}C")
        ax1.set_xlabel("time since workload start (s)")
        ax1.set_ylabel("temperature (C)", color="#c0392b")
        ax2 = ax1.twinx()
        ax2.plot(rel, np.array(pw) / 1000.0, color="#2c3e50", alpha=0.7, label="compute power")
        ax2.set_ylabel("compute-rail power (W)", color="#2c3e50")
        ax1.set_title(f"Thermal/power over time — {model} / {mode} / {cfg} (r{rep})")
        ax1.grid(alpha=0.3)
        _save(fig, plt, f"F3_thermal_{model}_{mode}_{cfg}_r{rep}")
 
 
def fig_thermal_compare(plt, np, model="complex", mode="MAXN"):
    """F4: overlay junction temp(t) for INT8 vs FP16 (same model/mode) — the punchline."""
    series = []
    for cfg in ["GPU-FP32", "GPU-FP16"]:
        hits = sorted(glob.glob(os.path.join(RESULTS_DIR, f"ts_{model}_{mode}_{cfg}_r*.csv")))
        if hits:
            rel, tj, _ = parse_timeseries(hits[0])  # first repeat as representative
            if rel:
                series.append((cfg, rel, tj))
    if len(series) < 2:
        print("[analyze] not enough ts files for thermal compare — skipping F4")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for cfg, rel, tj in series:
        ax.plot(rel, tj, label=cfg, color=CONFIG_COLORS.get(cfg))
    ax.set_xlabel("time since workload start (s)")
    ax.set_ylabel("junction temperature (C)")
    ax.set_title(f"Heat-up under load: INT8 vs FP16 — {model} / {mode}")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, plt, f"F4_thermal_compare_{model}_{mode}")
 
 
# ----------------------------------------------------------------------------- 
def write_master_table(agg, acc):
    """Joined mean+/-std table -> your deck's table slide."""
    rows = []
    for _, r in agg.iterrows():
        row = {
            "model": r["model"], "power_mode": r["power_mode"], "config": r["config"],
            "precision": r["precision"],
            "throughput_ips": f"{r['throughput_ips_mean']:.1f}+/-{r['throughput_ips_std']:.1f}",
            "latency_ms_p99": f"{r['latency_ms_p99_mean']:.2f}",
            "dyn_power_W": f"{r['dynamic_compute_mw_mean_mean']/1000:.2f}",
            "energy_mJ_per_inf": f"{r['energy_mj_per_inf_mean']:.2f}",
            "perf_per_watt": f"{r['perf_per_watt_ips_w_mean']:.2f}+/-{r['perf_per_watt_ips_w_std']:.2f}",
            "tj_peak_C": f"{r['tj_peak_c_mean']:.1f}",
        }
        if acc is not None:
            a = acc[(acc.model == r["model"]) & (acc.precision == r["precision"])]
            row["top1_%"] = f"{float(a['top1'].iloc[0])*100:.2f}" if not a.empty else ""
        rows.append(row)
    out = os.path.join(RESULTS_DIR, "master_table.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[analyze] wrote {out}")
 
 
# ----------------------------------------------------------------------------- 
def _save(fig, plt, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name + ".png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[analyze] {path}")
 
 
def main():
    global RESULTS_DIR, FIG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--fan-threshold", type=float, default=70.0,
                    help="draw the fan-on threshold line at this temp (match the harness)")
    ap.add_argument("--compare-model", default="complex")
    ap.add_argument("--compare-mode", default="15W")
    args = ap.parse_args()
 
    RESULTS_DIR = args.results_dir
    FIG_DIR = os.path.join(RESULTS_DIR, "figures")
 
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} not found — run characterize.py first")
    df = pd.read_csv(summary_path)
    agg = _agg(df)
 
    acc_path = os.path.join(RESULTS_DIR, "accuracy.csv")
    acc = pd.read_csv(acc_path) if os.path.exists(acc_path) else None
    if acc is None:
        print("[analyze] accuracy.csv missing — trade-off + accuracy column will be skipped")
 
    fig_perf_per_watt(agg, plt, np)
    fig_throughput(agg, plt, np)
    fig_tradeoff(agg, acc, plt, np)
    fig_thermal_timeseries(plt, np, fan_threshold=args.fan_threshold)
    fig_thermal_compare(plt, np, model=args.compare_model, mode=args.compare_mode)
    write_master_table(agg, acc)
    print("[analyze] done. Figures in", FIG_DIR)
 
 
if __name__ == "__main__":
    main()
