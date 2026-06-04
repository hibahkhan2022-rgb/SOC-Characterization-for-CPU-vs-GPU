import os
import sys
import glob
import shutil
import random
import tarfile
import argparse
import urllib.request
 
# Imagenette (320px) — public, hosted by fast.ai.
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
ARCHIVE = "imagenette2-320.tgz"
EXTRACT_DIR = "imagenette2-320"
 
# Imagenette WNID -> ImageNet-1k class index. These are the standard mappings; if your
# accuracy looks impossibly low for FP32, re-verify these against the torchvision class
# list (the model's metadata categories) before trusting INT8 deltas.
WNID_TO_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}
 
DATA_DIR = "data"
CALIB_DIR = os.path.join(DATA_DIR, "calib")
VAL_DIR = os.path.join(DATA_DIR, "val")
 
 
def _download(url, dest):
    if os.path.exists(dest):
        print(f"[fetch] {dest} already present, skipping download")
        return
    print(f"[fetch] downloading {url} ...")
    def _progress(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            sys.stdout.write(f"\r[fetch]   {pct}%")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, _progress)
    print("\n[fetch] download complete")
 
 
def _extract(archive, target):
    if os.path.isdir(target):
        print(f"[fetch] {target} already extracted, skipping")
        return
    print(f"[fetch] extracting {archive} ...")
    with tarfile.open(archive) as t:
        t.extractall(".")
    print("[fetch] extracted")
 
 
def build_calib(src_train, n_per_class, seed=0):
    """Flat folder of representative images sampled across all classes."""
    os.makedirs(CALIB_DIR, exist_ok=True)
    random.seed(seed)
    count = 0
    for wnid in WNID_TO_INDEX:
        cls_dir = os.path.join(src_train, wnid)
        imgs = glob.glob(os.path.join(cls_dir, "*.JPEG"))
        random.shuffle(imgs)
        for src in imgs[:n_per_class]:
            dst = os.path.join(CALIB_DIR, f"{wnid}_{os.path.basename(src)}")
            shutil.copy(src, dst)
            count += 1
    print(f"[calib] {count} images -> {CALIB_DIR}")
 
 
def build_val(src_val, n_per_class):
    """Labeled val set, folders named by ImageNet-1k INDEX (prepare_models expects this)."""
    total = 0
    for wnid, idx in WNID_TO_INDEX.items():
        src_cls = os.path.join(src_val, wnid)
        if not os.path.isdir(src_cls):
            print(f"[val][warn] missing {src_cls}", file=sys.stderr)
            continue
        dst_cls = os.path.join(VAL_DIR, str(idx))   # <-- folder named by integer index
        os.makedirs(dst_cls, exist_ok=True)
        imgs = sorted(glob.glob(os.path.join(src_cls, "*.JPEG")))[:n_per_class]
        for src in imgs:
            shutil.copy(src, os.path.join(dst_cls, os.path.basename(src)))
            total += 1
    print(f"[val] {total} labeled images across {len(WNID_TO_INDEX)} classes -> {VAL_DIR}")
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-per-class", type=int, default=30,
                    help="calibration images sampled per class (x10 classes)")
    ap.add_argument("--val-per-class", type=int, default=20,
                    help="labeled val images per class")
    ap.add_argument("--keep-archive", action="store_true",
                    help="don't delete the .tgz after extracting")
    args = ap.parse_args()
 
    os.makedirs(DATA_DIR, exist_ok=True)
    _download(IMAGENETTE_URL, ARCHIVE)
    _extract(ARCHIVE, EXTRACT_DIR)
 
    train = os.path.join(EXTRACT_DIR, "train")
    val = os.path.join(EXTRACT_DIR, "val")
    if not (os.path.isdir(train) and os.path.isdir(val)):
        print(f"[error] expected {train} and {val} after extract", file=sys.stderr)
        sys.exit(1)
 
    build_calib(train, args.calib_per_class)
    build_val(val, args.val_per_class)
 
    if not args.keep_archive and os.path.exists(ARCHIVE):
        os.remove(ARCHIVE)
 
    print("\n[done] data ready:")
    print(f"  calibration : {CALIB_DIR}/  (point prepare_models.py --calib-dir here)")
    print(f"  validation  : {VAL_DIR}/    (point prepare_models.py --imagenet-val here)")
    print("[note] calibration images must NOT overlap your val set (leakage). This script "
          "samples calib from train/ and val from val/, so they're disjoint by construction.")
 
 
if __name__ == "__main__":
    main()
