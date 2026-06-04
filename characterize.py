import os
import csv
import sys
import glob
import time
import json
import argparse
import threading
import statistics
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
 
# =============================================================================
# CONFIG  — VERIFY EVERY VALUE IN THIS BLOCK ON YOUR OWN BOARD BEFORE TRUSTING DATA
# =============================================================================
 
# --- nvpmodel power-mode IDs.  Run `sudo nvpmodel -q` and read the IDs off YOUR
#     board.  On the original Orin Nano these are commonly MAXN=0 and 7W=1, but
#     the "Super" board and different JetPack versions renumber them. DO NOT trust
#     these defaults blindly. ---
POWER_MODES = {
    "15W": 0,   # verified via /etc/nvpmodel.conf on this board
    "7W":  1,   # verified via /etc/nvpmodel.conf on this board
}
 
# --- Thermal protocol (tune to exactly what you mean) ---
START_TEMP_C        = 60.0   # cool to this (fan on) before each trial. 55C was below
                             # this board's fan-on floor -> every trial hit the timeout.
                             # Set to a few degrees above your fan-on idle temp (check
                             # `cat /sys/devices/virtual/thermal/thermal_zone*/temp`).
FAN_ON_THRESHOLD_C  = 70.0   # temperature whose crossing starts the fan-on countdown
FAN_ON_DELAY_S      = 120.0  # turn fan ON this many seconds after threshold is crossed
COOLDOWN_S          = 120.0  # cooldown (fan on, idle) after each trial
COOL_TIMEOUT_S      = 120.0  # give up cooling to START_TEMP_C after this long (warn),
                             # so a slightly-too-low target can't stall the whole run
 
# --- Timing ---
WARMUP_ITERS    = 30     # discarded (lazy init, allocation, clock ramp)
IDLE_SAMPLE_S   = 10.0   # measure idle power for this long before the workload
RUN_DURATION_S  = 180.0  # workload duration per trial; long enough to reach thermal
                         # steady state / observe throttle + fan-on recovery
N_REPEATS       = 5      # repeats per cell -> mean +/- std
POWER_SAMPLE_HZ = 10.0   # INA3221 sampling rate (Hz). Keep >> 1/inference_time.
 
# --- Fan control (paths vary by carrier board / L4T version; auto-discovery below
#     tries common locations, but confirm yours). ---
NVFANCONTROL_SVC = "nvfancontrol"            # systemd service to stop for manual control
FAN_PWM_GLOBS = [
    "/sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1",
    "/sys/class/hwmon/hwmon*/pwm1",
]
FAN_PWM_MAX = 255
 
# --- INA3221: rail labels we care about. VDD_IN = total module; VDD_CPU_GPU_CV =
#     combined compute rail on Orin Nano. Auto-discovered by label below. ---
INA_HWMON_GLOB = "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"
RAILS_OF_INTEREST = ["VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"]  # logs whichever exist
 
# --- Thermal zones: auto-discovered; we track the max ("junction") and per-zone. ---
THERMAL_ZONE_GLOB = "/sys/devices/virtual/thermal/thermal_zone*"
 
OUTPUT_DIR = "results"
 
 
# =============================================================================
# Low-level device interfaces
# =============================================================================
 
def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None
 
def _read_str(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None
 
def _sh(cmd: list) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
 
class PowerMode:
    """Sets nvpmodel mode and locks clocks for repeatability."""
    @staticmethod
    def set(mode_name: str) -> None:
        mode_id = POWER_MODES[mode_name]
        # MANUAL-MODE PASS: nvpmodel switching is DISABLED here because on this board
        # the 7W<->15W switch changes the online CPU core count and forces a REBOOT.
        # Set the power mode by hand before running (sudo nvpmodel -m <id>), confirm
        # with `nvpmodel -q`, then run this script for that ONE mode only. We just lock
        # clocks; we never call nvpmodel (which would reboot mid-run).
        print(f"[mode] assuming board is already in {mode_name} "
              f"(nvpmodel switching disabled for manual pass)", flush=True)
        try:
            subprocess.run(["jetson_clocks"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60)
        except Exception as e:
            print(f"[warn] jetson_clocks failed: {e}", file=sys.stderr)
 
 
class Fan:
    """Manual fan control: stop the auto service, drive pwm directly."""
    def __init__(self):
        self.pwm_path = self._find_pwm()
        if self.pwm_path is None:
            print("[warn] no fan pwm node found; fan control disabled", file=sys.stderr)
 
    @staticmethod
    def _find_pwm() -> Optional[str]:
        for g in FAN_PWM_GLOBS:
            hits = glob.glob(g)
            if hits:
                return hits[0]
        return None
 
    def take_manual_control(self):
        # Stopping the service is what actually lets manual pwm writes stick.
        try:
            _sh(["systemctl", "stop", NVFANCONTROL_SVC])
        except Exception:
            pass  # service may not exist on all images
 
    def release(self):
        try:
            _sh(["systemctl", "start", NVFANCONTROL_SVC])
        except Exception:
            pass
 
    def set(self, on: bool):
        if not self.pwm_path:
            return
        val = FAN_PWM_MAX if on else 0
        try:
            with open(self.pwm_path, "w") as f:
                f.write(str(val))
        except Exception as e:
            print(f"[warn] fan write failed: {e}", file=sys.stderr)
 
 
class Thermal:
    """Reads thermal zones; junction temp = max across zones (degrees C)."""
    def __init__(self):
        self.zones = []  # (name, temp_path)
        for z in sorted(glob.glob(THERMAL_ZONE_GLOB)):
            name = _read_str(os.path.join(z, "type")) or os.path.basename(z)
            tpath = os.path.join(z, "temp")
            if os.path.exists(tpath):
                self.zones.append((name, tpath))
 
    def read_all(self) -> dict:
        out = {}
        for name, tpath in self.zones:
            milli = _read_int(tpath)
            if milli is not None:
                out[name] = milli / 1000.0
        return out
 
    def tj(self) -> float:
        vals = self.read_all().values()
        return max(vals) if vals else float("nan")
 
 
class INA3221:
    """Auto-discovers INA3221 channels by rail label; reads V (mV) and I (mA),
    computes power (mW). On Orin Nano VDD_CPU_GPU_CV is the COMBINED compute rail."""
    def __init__(self):
        self.channels = {}  # rail_name -> (volt_path, curr_path)
        for hw in glob.glob(INA_HWMON_GLOB):
            for label_path in glob.glob(os.path.join(hw, "in*_label")):
                rail = _read_str(label_path)
                if rail in RAILS_OF_INTEREST:
                    idx = os.path.basename(label_path).replace("_label", "").replace("in", "")
                    volt = os.path.join(hw, f"in{idx}_input")     # bus voltage, mV
                    curr = os.path.join(hw, f"curr{idx}_input")   # current, mA
                    if os.path.exists(volt) and os.path.exists(curr):
                        self.channels[rail] = (volt, curr)
        if not self.channels:
            print("[warn] no INA3221 rails found; power logging disabled", file=sys.stderr)
 
    def sample_mw(self) -> dict:
        out = {}
        for rail, (vp, cp) in self.channels.items():
            mv = _read_int(vp)
            ma = _read_int(cp)
            if mv is not None and ma is not None:
                out[rail] = (mv * ma) / 1000.0   # mW
        return out
 
 
class PowerLogger(threading.Thread):
    """Background sampler: records (t, {rail: mW}, {zone: C}) at POWER_SAMPLE_HZ."""
    def __init__(self, ina: INA3221, thermal: Thermal, hz: float):
        super().__init__(daemon=True)
        self.ina, self.thermal = ina, thermal
        self.period = 1.0 / hz
        self._stop_event = threading.Event()  # NOT _stop: that shadows Thread._stop()
        self.samples = []  # list of dicts
 
    def run(self):
        while not self._stop_event.is_set():
            t = time.time()
            self.samples.append({
                "t": t,
                "power_mw": self.ina.sample_mw(),
                "temp_c": self.thermal.read_all(),
            })
            time.sleep(self.period)
 
    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)
 
    def window(self, t0: float, t1: float):
        return [s for s in self.samples if t0 <= s["t"] <= t1]
 
 
# =============================================================================
# Inference backends  — PLUG YOUR MODELS IN HERE
# =============================================================================
 
class InferenceBackend:
    name: str
    device: str      # "GPU" | "CPU"
    precision: str   # "INT8" | "FP16" | "FP32"
    def load(self): ...
    def infer_once(self): ...          # one inference on a fixed dummy/real input
    def infer_labeled(self, sample):   # for accuracy mode; return predicted class id
        raise NotImplementedError
 
 
class TensorRTBackend(InferenceBackend):
    """GPU inference from a prebuilt TensorRT engine.
 
    Build engines OFFLINE first, e.g.:
      FP16: trtexec --onnx=model.onnx --fp16 --saveEngine=model_fp16.engine
      INT8: trtexec --onnx=model.onnx --int8 --calib=calib.cache \
                    --saveEngine=model_int8.engine
    VERIFY precision actually applied with: trtexec ... --verbose  (read per-layer
    precision / 'Layer(...) -> Int8'). This is your defense against silent fallback.
    """
    def __init__(self, engine_path: str, precision: str, name: str):
        self.engine_path = engine_path
        self.precision = precision
        self.device = "GPU"
        self.name = name
        self._ctx = None
 
    def load(self):
        # TensorRT 10.x (JetPack 6.2) tensor-I/O API: name-based addressing +
        # execute_async_v3 on a stream. The old binding-index calls
        # (num_bindings/get_binding_shape/execute_v2) were REMOVED in TRT 10.
        import tensorrt as trt          # noqa
        import pycuda.driver as cuda    # noqa
        import pycuda.autoinit          # noqa
        import numpy as np
        self.cuda, self.np = cuda, np
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        # Allocate one host+device buffer per I/O tensor and bind device addresses
        # to tensor names. Assumes a single-input/single-output static-shape
        # classifier (ResNet-50 / MobileNetV3 fit). Extend the lists for multi-IO.
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))  # trt->numpy
            shape = self.engine.get_tensor_shape(name)
            size = int(np.prod([int(d) for d in shape]))
            host = cuda.pagelocked_empty(size, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            entry = {"name": name, "host": host, "dev": dev}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                host[:] = 0  # dummy until a real preprocessed sample is written
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)
        # convenience handles for the single-IO case (used by infer_labeled/accuracy)
        self.host_in, self.dev_in = self.inputs[0]["host"], self.inputs[0]["dev"]
        self.host_out, self.dev_out = self.outputs[0]["host"], self.outputs[0]["dev"]
 
    def infer_once(self):
        # Includes stream.synchronize() so timing this call measures true completion
        # (kernel launch alone would under-report latency).
        c = self.cuda
        c.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        c.memcpy_dtoh_async(self.host_out, self.dev_out, self.stream)
        self.stream.synchronize()
        return self.host_out
 
    def infer_labeled(self, sample):
        # TODO: preprocess `sample` into self.host_in, run, argmax the output.
        self.host_in[:] = sample.ravel()
        out = self.infer_once()
        return int(self.np.argmax(out))
 
 
class ORTCpuBackend(InferenceBackend):
    """CPU baseline via ONNX Runtime (CPUExecutionProvider). Effectively FP32.
    This is a DIFFERENT runtime than the GPU path -> runtime confound; state it."""
    def __init__(self, onnx_path: str, name: str):
        self.onnx_path = onnx_path
        self.device = "CPU"
        self.precision = "FP32"
        self.name = name
 
    def load(self):
        import onnxruntime as ort
        import numpy as np
        so = ort.SessionOptions()
        # Pin threads for repeatability; record this in methodology.
        so.intra_op_num_threads = 0  # 0 = let ORT decide; set explicitly if you want control
        self.sess = ort.InferenceSession(
            self.onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.inp = self.sess.get_inputs()[0]
        self.np = np
        self.dummy = np.zeros([d if isinstance(d, int) else 1 for d in self.inp.shape],
                              dtype=np.float32)
 
    def infer_once(self):
        return self.sess.run(None, {self.inp.name: self.dummy})
 
    def infer_labeled(self, sample):
        out = self.sess.run(None, {self.inp.name: sample.astype(self.np.float32)})
        return int(self.np.argmax(out[0]))
 
 
# =============================================================================
# Trial orchestration
# =============================================================================
 
@dataclass
class TrialResult:
    model: str
    power_mode: str
    config: str
    device: str
    precision: str
    repeat: int
    # performance
    latency_ms_mean: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p99: float = 0.0
    throughput_ips: float = 0.0
    iters: int = 0
    # power (mW), idle-subtracted dynamic on the combined compute rail
    idle_compute_mw: float = 0.0
    active_compute_mw_mean: float = 0.0
    dynamic_compute_mw_mean: float = 0.0
    total_module_mw_mean: float = 0.0
    energy_mj_per_inf: float = 0.0      # dynamic compute energy per inference
    perf_per_watt_ips_w: float = 0.0
    # thermal
    tj_start_c: float = 0.0
    tj_peak_c: float = 0.0
    throttled: bool = False             # set if tj exceeded throttle temp during run
 
 
def cool_to_start(thermal: Thermal, fan: Fan):
    fan.set(True)
    t0 = time.time()
    while True:
        tj = thermal.tj()
        if tj <= START_TEMP_C:
            return tj
        if time.time() - t0 > COOL_TIMEOUT_S:
            print(f"[warn] could not cool to {START_TEMP_C}C (stuck at {tj:.1f}C). "
                  f"Ambient too warm? Proceeding — flag this trial.", file=sys.stderr)
            return tj
        time.sleep(2.0)
 
 
def measure_idle(logger: PowerLogger, seconds: float) -> float:
    t0 = time.time()
    time.sleep(seconds)
    win = logger.window(t0, time.time())
    vals = [s["power_mw"].get("VDD_CPU_GPU_CV") for s in win
            if s["power_mw"].get("VDD_CPU_GPU_CV") is not None]
    return statistics.mean(vals) if vals else 0.0
 
 
def run_trial(backend: InferenceBackend, model: str, power_mode: str, config: str,
              repeat: int, ina: INA3221, thermal: Thermal, fan: Fan) -> TrialResult:
    res = TrialResult(model=model, power_mode=power_mode, config=config,
                      device=backend.device, precision=backend.precision, repeat=repeat)
 
    # 1) thermal precondition (fan ON, cool to 55C)
    res.tj_start_c = cool_to_start(thermal, fan)
 
    # 2) start background power/thermal logging
    logger = PowerLogger(ina, thermal, POWER_SAMPLE_HZ)
    logger.start()
 
    # 3) idle baseline at this power mode (fan still on, no workload)
    res.idle_compute_mw = measure_idle(logger, IDLE_SAMPLE_S)
 
    # 4) warm-up (discarded)
    for _ in range(WARMUP_ITERS):
        backend.infer_once()
 
    # 5) fan OFF, begin timed workload; turn fan back on FAN_ON_DELAY_S after
    #    tj first crosses FAN_ON_THRESHOLD_C.
    fan.set(False)
    fan_back_on = False
    threshold_crossed_at = None
    latencies = []
    t_run0 = time.time()
    while time.time() - t_run0 < RUN_DURATION_S:
        t_i = time.perf_counter()
        backend.infer_once()
        latencies.append((time.perf_counter() - t_i) * 1000.0)  # ms
 
        tj_now = thermal.tj()
        if tj_now >= FAN_ON_THRESHOLD_C and threshold_crossed_at is None:
            threshold_crossed_at = time.time()
            res.throttled = True  # crossing your chosen threshold; refine vs HW throttle pt
        if (threshold_crossed_at is not None and not fan_back_on
                and time.time() - threshold_crossed_at >= FAN_ON_DELAY_S):
            fan.set(True)
            fan_back_on = True
    t_run1 = time.time()
    logger.stop()
    if not fan_back_on:
        fan.set(True)
 
    # 6) reduce performance
    res.iters = len(latencies)
    res.latency_ms_mean = statistics.mean(latencies)
    res.latency_ms_p50 = statistics.median(latencies)
    res.latency_ms_p99 = sorted(latencies)[int(0.99 * len(latencies)) - 1]
    res.throughput_ips = res.iters / (t_run1 - t_run0)
 
    # 7) reduce power over the workload window (combined compute rail)
    win = logger.window(t_run0, t_run1)
    comp = [s["power_mw"].get("VDD_CPU_GPU_CV") for s in win
            if s["power_mw"].get("VDD_CPU_GPU_CV") is not None]
    tot = [s["power_mw"].get("VDD_IN") for s in win
           if s["power_mw"].get("VDD_IN") is not None]
    res.active_compute_mw_mean = statistics.mean(comp) if comp else 0.0
    res.total_module_mw_mean = statistics.mean(tot) if tot else 0.0
    res.dynamic_compute_mw_mean = max(0.0, res.active_compute_mw_mean - res.idle_compute_mw)
 
    # energy per inference (mJ) = dynamic power (W) * latency (s)  -> use mean latency
    dyn_w = res.dynamic_compute_mw_mean / 1000.0
    res.energy_mj_per_inf = dyn_w * (res.latency_ms_mean / 1000.0) * 1000.0
    res.perf_per_watt_ips_w = (res.throughput_ips / dyn_w) if dyn_w > 0 else 0.0
 
    # thermal peak over the run
    tj_series = [max(s["temp_c"].values()) for s in win if s["temp_c"]]
    res.tj_peak_c = max(tj_series) if tj_series else float("nan")
 
    # 8) save the time-series for this trial (for your throttle/recovery plots)
    _save_timeseries(logger, model, power_mode, config, repeat)
    return res
 
 
def _save_timeseries(logger, model, power_mode, config, repeat):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fn = os.path.join(OUTPUT_DIR, f"ts_{model}_{power_mode}_{config}_r{repeat}.csv")
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "rail", "mW"] + ["zone", "C"])
        for s in logger.samples:
            for rail, mw in s["power_mw"].items():
                w.writerow([s["t"], rail, mw, "", ""])
            for zone, c in s["temp_c"].items():
                w.writerow([s["t"], "", "", zone, c])
 
 
# =============================================================================
# Backend registry  — EDIT THESE PATHS to your engines/model
# =============================================================================
 
def build_backends():
    """Return {(model, config): backend}. Build TRT engines offline first.
    NOTE: GPU-INT8 dropped — TRT 10.3 on this Orin Nano has no INT8 tactic for the
    ResNet maxpool (documented limitation). Precision axis is FP32 -> FP16 instead;
    still a real precision/datapath characterization (FP16 gains perf/watt at zero
    accuracy cost here). To restore INT8 later: add the GPU-INT8 entries back and
    resolve the pooling build (onnxsim or a maxpool workaround)."""
    return {
        ("complex", "GPU-FP32"): TensorRTBackend("engines/complex_fp32.engine", "FP32", "complex_fp32"),
        ("complex", "GPU-FP16"): TensorRTBackend("engines/complex_fp16.engine", "FP16", "complex_fp16"),
        ("complex", "CPU"):      ORTCpuBackend("models/complex.onnx", "complex_cpu"),
        ("light",   "GPU-FP32"): TensorRTBackend("engines/light_fp32.engine", "FP32", "light_fp32"),
        ("light",   "GPU-FP16"): TensorRTBackend("engines/light_fp16.engine", "FP16", "light_fp16"),
        ("light",   "CPU"):      ORTCpuBackend("models/light.onnx", "light_cpu"),
    }
 
MODELS  = ["complex", "light"]
PMODES  = ["7W"]   # MANUAL 7W PASS: 15W already collected separately; merge after
CONFIGS = ["GPU-FP32", "GPU-FP16", "CPU"]
 
 
# =============================================================================
# Main
# =============================================================================
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--dry-run", action="store_true",
                    help="check device discovery + backends without running trials")
    ap.add_argument("--smoke", action="store_true",
                    help="fast single-cell validation run: 1 cell, short timings, no "
                         "cool-wait. Confirms the full loop + CSV before a real run.")
    args = ap.parse_args()
 
    # Smoke test: override timing globals and shrink the matrix to ONE cell so the
    # whole pipeline (mode set -> idle -> warmup -> run -> log -> cooldown -> CSV)
    # exercises in ~1 minute. Nothing to revert: real runs just omit --smoke.
    global RUN_DURATION_S, COOLDOWN_S, IDLE_SAMPLE_S, START_TEMP_C, WARMUP_ITERS, \
        FAN_ON_THRESHOLD_C
    if args.smoke:
        RUN_DURATION_S, COOLDOWN_S, IDLE_SAMPLE_S = 15.0, 10.0, 3.0
        WARMUP_ITERS = 5
        START_TEMP_C = 200.0        # effectively skip the cool-to-55 wait
        FAN_ON_THRESHOLD_C = 999.0  # don't trip the fan-on logic during the smoke run
        args.repeats = 1
        models, pmodes, configs = MODELS[:1], PMODES[:1], CONFIGS[:1]
        print("[smoke] one cell, short timings — validating loop + outputs only "
              "(NOT real data)")
    else:
        models, pmodes, configs = MODELS, PMODES, CONFIGS
 
    if os.geteuid() != 0:
        print("[warn] not root: nvpmodel/fan/clock control will likely fail. "
              "Re-run with sudo.", file=sys.stderr)
 
    ina, thermal, fan = INA3221(), Thermal(), Fan()
    print(f"[info] INA rails: {list(ina.channels.keys())}")
    print(f"[info] thermal zones: {[z[0] for z in thermal.zones]}")
    print(f"[info] fan pwm: {fan.pwm_path}")
 
    backends = build_backends()
    if args.dry_run:
        print("[info] dry run OK; verify the above discovery matches your board.")
        return
 
    fan.take_manual_control()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "summary.csv")
    rows = []
 
    def _flush():
        # rewrite the full summary after each trial so an interrupt keeps completed work
        if rows:
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
 
    try:
        for power_mode in pmodes:
            PowerMode.set(power_mode)
            for model in models:
                for config in configs:
                    backend = backends[(model, config)]
                    backend.load()
                    for r in range(args.repeats):
                        print(f"[run] {power_mode} | {model} | {config} | repeat {r}")
                        res = run_trial(backend, model, power_mode, config, r,
                                        ina, thermal, fan)
                        rows.append(asdict(res))
                        _flush()  # persist immediately
                        # cooldown between trials
                        fan.set(True)
                        time.sleep(COOLDOWN_S)
    except KeyboardInterrupt:
        print(f"\n[stopped] interrupted after {len(rows)} trials; "
              f"summary.csv has those rows.", file=sys.stderr)
    finally:
        fan.release()       # hand control back to nvfancontrol no matter what
        _flush()
    print(f"[done] wrote {out_csv} ({len(rows)} trials) + per-trial time series.")
 
 
if __name__ == "__main__":
    main()
