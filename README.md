# Edge AI Inference Characterization — Jetson Orin Nano

A power / performance / thermal **characterization** of AI inference across compute
datapaths and numerical precisions on a power- and thermally-constrained edge SoC.

> **Thesis.** Not "which config is fastest," but: *given a workload and a power budget,
> which compute datapath should you use, and what does it cost?* The deliverable is a
> small **decision surface** over (model scale × power budget × precision), reported with
> the accuracy cost of quantization and the sustained-vs-peak thermal behavior.

This is the same discipline as post-silicon power/performance characterization: measure a
part's real operating envelope through its on-die telemetry, verify behavior against
claims, and report findings with stated limitations and variance.

---

## Experiment matrix

`2 models × 2 power modes × 3 configs = 12 cells`, each repeated `N_REPEATS` times.

| Axis | Levels |
|---|---|
| Model | complex (ResNet-50), lightweight (MobileNetV3) |
| Power mode | MAXN (Super, ~25W on JetPack 6.2), 7W |
| Config | GPU-INT8 (TensorRT), GPU-FP16 (TensorRT), CPU-FP32 (ONNX Runtime) |

FP32 GPU engines are also built (accuracy reference / clean precision sweep).

---

## Pipeline & run order

```
fetch_data.py  ->  prepare_models.py  ->  characterize.py  ->  analyze.py
   (data)            (engines+acc)          (measure)           (figures)
```

```bash
# 0. confirm the environment (do this FIRST — see "Verify before trusting data")
sudo nvpmodel -q
python3 -c "import tensorrt as trt; print(trt.__version__)"   # expect 10.3

# 1. data: download Imagenette, build calib set + index-labeled val set
python3 fetch_data.py --calib-per-class 30 --val-per-class 20

# 2. models: ONNX export -> TensorRT engines (FP32/FP16/INT8) + accuracy pass
python3 prepare_models.py all --calib-dir data/calib --imagenet-val data/val --verbose

# 3. PROVE the INT8 engine is really INT8 (run for each int8 engine)
python3 prepare_models.py verify --engine engines/complex_int8.engine
python3 prepare_models.py verify --engine engines/light_int8.engine

# 4. measure: walk the 12-cell matrix under the thermal protocol  (root needed)
sudo python3 characterize.py            # use --dry-run first to confirm discovery

# 5. analyze: CSVs -> figures + master table
python3 analyze.py --fan-threshold 70
```

Outputs land in `results/` (`summary.csv`, `ts_*.csv`, `accuracy.csv`,
`master_table.csv`, `figures/*.png`).

---

## Directory layout

```
fetch_data.py        # download + arrange data (stdlib only)
prepare_models.py    # ONNX export, TRT engine build w/ INT8 calibration, verify, accuracy
characterize.py      # the measurement harness (power mode, fan, thermal, INA3221, timing)
analyze.py           # CSVs -> deck figures + joined master table
requirements.txt
README.md
data/   calib/  val/<class_index>/      # inputs (built by fetch_data.py)
models/ <name>.onnx                      # exported models
engines/ <name>_<prec>.engine            # built TRT engines
results/ summary.csv  accuracy.csv  ts_*.csv  master_table.csv  figures/
```

---

## Verify before trusting data (the things that silently corrupt results)

1. **Power-mode IDs.** Run `sudo nvpmodel -q` and set `POWER_MODES` in `characterize.py`
   to the *actual* IDs on this board. On JetPack 6.2 "MAXN" is MAXN-Super; the IDs are
   **not** guaranteed to be the 0/1 defaults.
2. **INT8 is actually INT8.** Run the `verify` step and confirm the heavy conv/matmul
   layers are INT8, not silently fallen back to FP16/FP32.
3. **Calibration ≠ validation set.** `fetch_data.py` samples calib from `train/` and val
   from `val/` (disjoint by construction) — keep it that way; overlap = leakage.
4. **Device discovery.** `sudo python3 characterize.py --dry-run` must list the INA3221
   rails (incl. `VDD_CPU_GPU_CV`, `VDD_IN`), thermal zones, and a fan PWM path. If any are
   empty, fix the paths before a real run.
5. **Thermal constants match.** The `--fan-threshold` in `analyze.py` must equal
   `FAN_ON_THRESHOLD_C` in `characterize.py`, or the plotted threshold line lies.

---

## Known limitations (state these; they read as maturity, not weakness)

- **Combined power rail.** On the Orin Nano, `VDD_CPU_GPU_CV` reports CPU+GPU power
  *combined*. Per-engine power isolation isn't possible from the rail alone; attribution
  here is by experimental design + idle subtraction, with acknowledged residual coupling.
  (An AGX Orin exposes separate VDD_GPU_SOC / VDD_CPU_CV rails.)
- **CPU runtime confound.** The CPU baseline runs on ONNX Runtime, the GPU paths on
  TensorRT. The CPU-vs-GPU comparison is therefore hardware+stack, not pure hardware.
- **CPU precision.** CPU runs are effectively FP32; labeled as such, not compared as
  equal precision to GPU INT8/FP16.
- **Accuracy on a 10-class subset (Imagenette).** Absolute accuracy is easy/high; the
  valid signal is the *relative* FP32→FP16→INT8 delta (the cost of quantization),
  measured on identical images. Use real ImageNet-1k val for absolute numbers.
- **No DLA.** The Orin Nano has no usable DLA; the specialized-accelerator trade-off is
  studied via the GPU Tensor-core (INT8) datapath instead. On Orin NX / AGX Orin the same
  methodology would extend to the DLA.
- **Two points, not curves.** 2 models and 2 power modes give *contrasts*, not trends —
  phrase conclusions as "for these regimes," not "as X increases."
- **Throughput-over-time not logged.** Time-series captures power+temperature; the
  throughput-during-throttle curve needs a small harness addition (timestamped latencies).

---

## Run record  (fill in for EACH data-collection session — this is your reproducibility)

```
Date / time         :
Board               : Jetson Orin Nano ( 4GB / 8GB / Super )  <-- circle
JetPack version     : 6.2.1            (apt-cache show nvidia-jetpack | grep Version)
TensorRT / CUDA     : 10.3 / 12.6
nvpmodel modes used : MAXN id = ___ , 7W id = ___   (from `nvpmodel -q`)
jetson_clocks       : on / off
Cooling             : stock fan / heatsink / other
Ambient temp        : ___ C        (affects cool-to-55 time and throttle onset)
Thermal constants   : START=55C  FAN_ON_THRESHOLD=70C  FAN_ON_DELAY=120s  COOLDOWN=300s
Run duration / reps : RUN_DURATION_S=___  N_REPEATS=___
Models              : complex=ResNet-50  light=MobileNetV3
Calib / val source  : Imagenette (calib-per-class=__, val-per-class=__)
Notes / anomalies   :
```
