import os
import csv
import glob
import argparse
 
# Heavy/device-only imports (torch, tensorrt, pycuda, onnxruntime) are done LAZILY
# inside the functions that need them, so `--help` and syntax checks work anywhere.
 
MODELS = {
    # logical name -> torchvision constructor name + weights enum
    "complex": ("resnet50", "ResNet50_Weights"),
    "light":   ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
}
INPUT_SHAPE = (1, 3, 224, 224)   # NCHW; static batch=1 (matches single-stream latency)
CALIB_BATCHES = 100              # ~100 batches of representative images is plenty
ONNX_DIR, ENGINE_DIR, RESULT_DIR = "models", "engines", "results"
 
 
# ----------------------------------------------------------------------------- 
# Preprocessing — MUST be identical across calibration, accuracy, and the harness's
# real inputs, or your numbers aren't comparable.
# -----------------------------------------------------------------------------
def imagenet_preprocess(pil_img, np):
    import numpy as _np  # noqa
    img = pil_img.convert("RGB").resize((256, 256))
    # center crop 224
    left = (256 - 224) // 2
    img = img.crop((left, left, left + 224, left + 224))
    x = (np.asarray(img).astype(np.float32) / 255.0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None]  # HWC -> NCHW, add batch
    return np.ascontiguousarray(x)
 
 
# ----------------------------------------------------------------------------- 
# export: torchvision -> ONNX
# -----------------------------------------------------------------------------
def cmd_export(args):
    import torch
    import torchvision.models as tvm
    os.makedirs(ONNX_DIR, exist_ok=True)
    for name in (args.models or MODELS.keys()):
        ctor_name, _ = MODELS[name]
        model = getattr(tvm, ctor_name)(weights="DEFAULT").eval()
        dummy = torch.zeros(*INPUT_SHAPE)
        out = os.path.join(ONNX_DIR, f"{name}.onnx")
        torch.onnx.export(
            model, dummy, out,
            input_names=["input"], output_names=["logits"],
            opset_version=17, do_constant_folding=True,
            dynamo=False,   # legacy TorchScript exporter: conventional ONNX graph that
                            # TRT's INT8 path builds cleanly (the new dynamo export
                            # produced a max_pool2d node with no INT8 implementation).
                            # Also single-file (no .onnx.data sidecar) at opset 17.
            # static shapes (no dynamic axes): simplest, matches batch=1 study.
        )
        print(f"[export] {name} ({ctor_name}) -> {out}")
 
 
# ----------------------------------------------------------------------------- 
# INT8 calibrator
# -----------------------------------------------------------------------------
def make_calibrator(calib_dir, cache_path):
    import tensorrt as trt
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa
    from PIL import Image
 
    class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.cache_path = cache_path
            self.files = sorted(glob.glob(os.path.join(calib_dir, "*")))[: CALIB_BATCHES]
            if not self.files:
                raise FileNotFoundError(f"no calibration images in {calib_dir}")
            self.idx = 0
            nbytes = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
            self.dev = cuda.mem_alloc(nbytes)
 
        def get_batch_size(self):
            return INPUT_SHAPE[0]
 
        def get_batch(self, names):
            if self.idx >= len(self.files):
                return None  # signals end of calibration data
            img = Image.open(self.files[self.idx]); self.idx += 1
            batch = imagenet_preprocess(img, np)
            cuda.memcpy_htod(self.dev, np.ascontiguousarray(batch, dtype=np.float32))
            return [int(self.dev)]
 
        def read_calibration_cache(self):
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    return f.read()
            return None
 
        def write_calibration_cache(self, cache):
            with open(self.cache_path, "wb") as f:
                f.write(cache)
 
    return ImageEntropyCalibrator()
 
 
# ----------------------------------------------------------------------------- 
# build: ONNX -> TensorRT engine (version-guarded for TRT 8.6 and 10.x)
# -----------------------------------------------------------------------------
def cmd_build(args):
    import tensorrt as trt
    os.makedirs(ENGINE_DIR, exist_ok=True)
 
    targets = args.models or MODELS.keys()
    precisions = args.precisions or ["fp32", "fp16", "int8"]
    for name in targets:
        onnx_path = os.path.join(ONNX_DIR, f"{name}.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"{onnx_path} missing — run `export` first")
        for prec in precisions:
            _build_one(trt, name, onnx_path, prec, args)
 
 
def _build_one(trt, name, onnx_path, prec, args):
    # VERBOSE logger so the build log carries per-layer precision (your proof).
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
 
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file (NOT parse(bytes)) so TRT resolves external-data weight files
    # (e.g. <name>.onnx.data) relative to the model's directory. Torch 2.11's exporter
    # externalizes weights, so parsing raw bytes would lose the path and fail to find them.
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("[build][onnx-error]", parser.get_error(i))
        raise RuntimeError(f"failed to parse {onnx_path}")
 
    config = builder.create_builder_config()
    # DETAILED verbosity is REQUIRED for the `verify` step's engine inspector to
    # report per-layer precision on TRT 10. Without it you get layer names only.
    if hasattr(config, "profiling_verbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # workspace pool: API differs across TRT versions
    if hasattr(config, "set_memory_pool_limit"):                  # TRT 8.4+/10.x
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    elif hasattr(config, "max_workspace_size"):                   # older TRT
        config.max_workspace_size = 2 << 30
 
    calibrator = None
    if prec == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif prec == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # allow FP16 for non-INT8 layers
        # On this TRT 10.3 / Orin Nano combo, POOLING layers have no INT8 tactic
        # ("could not find any implementation ... Time: inf") and won't fall back on
        # their own. Pin pooling to FP16 and tell TRT to honor the constraint; the
        # heavy conv/gemm layers still run INT8 (the real INT8 deployment pattern).
        if hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        pinned = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type == trt.LayerType.POOLING:
                layer.precision = trt.DataType.HALF
                pinned += 1
        print(f"[build]   pinned {pinned} pooling layer(s) to FP16 for INT8 build")
        cache = os.path.join(ENGINE_DIR, f"{name}_int8_calib.cache")
        calibrator = make_calibrator(args.calib_dir, cache)
        config.int8_calibrator = calibrator
    # fp32: no flags (default)
 
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed: {name} {prec}")
    out = os.path.join(ENGINE_DIR, f"{name}_{prec}.engine")
    with open(out, "wb") as f:
        f.write(serialized)
    print(f"[build] {name} {prec.upper()} -> {out}")
 
 
# ----------------------------------------------------------------------------- 
# verify: per-layer precision of a built engine
# -----------------------------------------------------------------------------
def cmd_verify(args):
    import tensorrt as trt
    import json
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    insp = engine.create_engine_inspector()
    info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    try:
        data = json.loads(info)
        layers = data.get("Layers", data if isinstance(data, list) else [])
        counts = {}
        for L in layers:
            p = (L.get("Precision") or L.get("precision") or "?")
            counts[p] = counts.get(p, 0) + 1
        print(f"[verify] {args.engine}")
        for p, c in sorted(counts.items()):
            print(f"         {p:>6}: {c} layers")
        if "INT8" in str(counts) or "Int8" in str(counts):
            print("         -> INT8 layers present (good). Confirm the heavy conv/gemm "
                  "layers are among them, not just a token few.")
        else:
            print("         -> NO INT8 layers found. If this is the int8 engine, "
                  "calibration/precision did not apply — investigate before trusting data.")
    except Exception:
        # Fallback: dump raw so you can eyeball it.
        print(info)
 
 
# ----------------------------------------------------------------------------- 
# accuracy: reuse the harness's inference backends (single code path)
# -----------------------------------------------------------------------------
def cmd_accuracy(args):
    import numpy as np
    from PIL import Image
    # import the SAME backends the measurement harness uses
    from characterize import TensorRTBackend, ORTCpuBackend
 
    # labeled val set layout: <imagenet_val>/<class_idx>/<image>.jpg
    samples = []  # (filepath, label_int)
    for cls_dir in sorted(glob.glob(os.path.join(args.imagenet_val, "*"))):
        if not os.path.isdir(cls_dir):
            continue
        try:
            label = int(os.path.basename(cls_dir))
        except ValueError:
            continue
        for img in sorted(glob.glob(os.path.join(cls_dir, "*")))[: args.per_class]:
            samples.append((img, label))
    if not samples:
        raise FileNotFoundError(
            f"no labeled images under {args.imagenet_val} "
            f"(expected <val>/<class_idx>/<img>)")
 
    os.makedirs(RESULT_DIR, exist_ok=True)
    rows = []
    for name in (args.models or MODELS.keys()):
        backends = {
            "FP32": TensorRTBackend(os.path.join(ENGINE_DIR, f"{name}_fp32.engine"), "FP32", f"{name}_fp32"),
            "FP16": TensorRTBackend(os.path.join(ENGINE_DIR, f"{name}_fp16.engine"), "FP16", f"{name}_fp16"),
            "INT8": TensorRTBackend(os.path.join(ENGINE_DIR, f"{name}_int8.engine"), "INT8", f"{name}_int8"),
            "CPU_FP32": ORTCpuBackend(os.path.join(ONNX_DIR, f"{name}.onnx"), f"{name}_cpu"),
        }
        for prec, backend in backends.items():
            try:
                backend.load()
            except Exception as e:
                print(f"[accuracy][skip] {name} {prec}: {e}")
                continue
            top1 = top5 = 0
            for path, label in samples:
                x = imagenet_preprocess(Image.open(path), np)
                # Get the full output vector (need it for top-5) from whichever backend.
                if hasattr(backend, "host_in"):          # TensorRT GPU backend
                    backend.host_in[:] = x.ravel()
                    out = np.asarray(backend.infer_once()).ravel()
                else:                                     # ONNX Runtime CPU backend
                    res = backend.sess.run(None, {backend.inp.name: x.astype(np.float32)})
                    out = np.asarray(res[0]).ravel()
                order = out.argsort()[::-1]
                if order[0] == label:
                    top1 += 1
                if label in order[:5]:
                    top5 += 1
            n = len(samples)
            row = {"model": name, "precision": prec,
                   "n": n, "top1": top1 / n, "top5": top5 / n}
            rows.append(row)
            print(f"[accuracy] {name:8} {prec:9} top1={row['top1']:.4f} top5={row['top5']:.4f} (n={n})")
 
    out_csv = os.path.join(RESULT_DIR, "accuracy.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "precision", "n", "top1", "top5"])
        w.writeheader(); w.writerows(rows)
    print(f"[accuracy] wrote {out_csv}")
    print("[note] the FP32->FP16->INT8 accuracy deltas ARE your 'cost of quantization' "
          "story. Pair them with the perf/watt gains from the harness.")
 
 
# ----------------------------------------------------------------------------- 
def cmd_all(args):
    cmd_export(args)
    cmd_build(args)
    if args.imagenet_val:
        cmd_accuracy(args)
    else:
        print("[all] skipped accuracy (no --imagenet-val given)")
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
 
    def add_common(p):
        p.add_argument("--models", nargs="*", choices=list(MODELS.keys()),
                       help="subset of models (default: all)")
 
    pe = sub.add_parser("export"); add_common(pe); pe.set_defaults(func=cmd_export)
 
    pb = sub.add_parser("build"); add_common(pb)
    pb.add_argument("--precisions", nargs="*", choices=["fp32", "fp16", "int8"])
    pb.add_argument("--calib-dir", default="data/calib",
                    help="folder of representative images for INT8 calibration")
    pb.add_argument("--verbose", action="store_true",
                    help="VERBOSE build log (read it to confirm per-layer precision)")
    pb.set_defaults(func=cmd_build)
 
    pv = sub.add_parser("verify")
    pv.add_argument("--engine", required=True)
    pv.set_defaults(func=cmd_verify)
 
    pa = sub.add_parser("accuracy"); add_common(pa)
    pa.add_argument("--imagenet-val", required=True,
                    help="labeled val set: <val>/<class_idx>/<img>.jpg")
    pa.add_argument("--per-class", type=int, default=5,
                    help="images per class to evaluate (keep small for a quick read)")
    pa.set_defaults(func=cmd_accuracy)
 
    pall = sub.add_parser("all"); add_common(pall)
    pall.add_argument("--precisions", nargs="*", choices=["fp32", "fp16", "int8"])
    pall.add_argument("--calib-dir", default="data/calib")
    pall.add_argument("--imagenet-val", default=None)
    pall.add_argument("--per-class", type=int, default=5)
    pall.add_argument("--verbose", action="store_true")
    pall.set_defaults(func=cmd_all)
 
    args = ap.parse_args()
    args.func(args)
 
 
if __name__ == "__main__":
    main()
 
