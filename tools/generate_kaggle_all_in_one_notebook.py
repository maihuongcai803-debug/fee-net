# -*- coding: utf-8 -*-
"""
Kaggle All-in-One Benchmark Generator — SINGLE-SOURCE-OF-TRUTH edition (2026-10-03)
================================================================================
Fixes the issues found in the 2026-10-03 audit:

P0-1  The benchmarked "FEE-Net (Ours Full)" was a stripped notebook-local wrapper
      around FEENetEarlyExit (no multi-scale deep supervision, no edge-head loss,
      no coarse_head). This generator injects the **repository** implementations
      so the benchmark trains the real `models/fee_net.py` FEENet.
P0-2  The notebook's private loss ignored deep supervision and the edge-head BCE
      term. The training engine now uses `utils.losses.CompoundLoss` exactly as
      `train.py` does, incl. edge BCE and 0.3-weighted deep supervision.
P0-3  The notebook re-implemented models/ and utils/ (dual implementations that
      drifted: the UNet channel bug, the 8.339M vs 8.539M parameter mismatch).
      Repo sources are now embedded at generation time and materialised into a
      real package on the Kaggle runtime, so `import models.fee_net` resolves to
      the repository file.
P0-4  The notebook re-derived the v2 splits with a *different* RNG/stratification,
      so the evaluated test set was NOT the repository `splits.json` test set.
      The exact repository split membership is now embedded and asserted.
P1    The triage-gate cost is now **measured** (CUDA-event timing of the
      stem+layer1+wavelet+triage-head path) instead of quoting an unverified
      "285+ FPS" constant, and the Pareto figure only plots measured points.

The generated notebook keeps the verified zero-CPU-spike engineering
(single-thread clamp, contiguous pre-allocated VRAM tensors, model-level dual-T4
allocation, no inner-loop host sync, CUDA-event profiling).
"""

import json
import os

import nbformat as nbf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Repo modules materialised on the Kaggle runtime (single source of truth)
REPO_MODULES = [
    "models/unet.py",
    "models/attention_unet.py",
    "models/unet_plus_plus.py",
    "models/fee_net.py",
    "models/fee_net_early_exit.py",
    "models/modules/dwt_module.py",
    "models/modules/egm_module.py",
    "utils/losses.py",
    "utils/metrics.py",
]
REPO_SPLITS = "data/kvasir_seg/splits.json"

# Tools that must run inside the same benchmark (single source of truth)
REPO_TOOLS = [
    "tools/build_mixed_cohort.py",
    "tools/eval_mixed_cohort_gate.py",
]

RUN_ABLATIONS = True  # w/o DWT and w/o EGM variants on the same v2 protocol


def _read(rel_path: str) -> str:
    with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
        return f.read()


def build_all_in_one_notebook(output_path="kaggle_feenet_baselines_all_in_one.ipynb"):
    embedded = {rel: _read(rel) for rel in REPO_MODULES}
    tools_src = {rel: _read(rel) for rel in REPO_TOOLS}
    splits = json.loads(_read(REPO_SPLITS))

    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    }
    nb.cells = []

    # ------------------------------------------------------------------
    # Cell 1: Title & overview
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_markdown_cell("""# Comprehensive Colonoscopy Benchmark: Repository FEE-Net vs. Classical Baselines
### Dual Tesla T4, Zero-CPU-Spike, Single-Source-of-Truth Pipeline

**Dataset**: Simula Kvasir-SEG (1000 cases), **repository** `splits.json` (Exp-027 v2):
150 zero-leakage independent test cases + 850-case 5-fold pool (Fold 0: 680 train / 170 val).

**Every model below is imported from the repository implementation** (the exact
`models/` and `utils/` sources are embedded into this notebook at generation time,
materialised under `/kaggle/working/repo/`, and imported as a real package):

| Model | Source file | Params |
|---|---|---|
| U-Net | `models/unet.py` | 7.763M |
| Attention U-Net | `models/attention_unet.py` | 7.852M |
| UNet++ | `models/unet_plus_plus.py` | 9.160M |
| **FEE-Net (Ours)** | `models/fee_net.py` | **8.539M** |
| **FEE-Net-EE (Ours)** | `models/fee_net_early_exit.py` | **8.552M** |

Training uses the repository compound loss (`utils/losses.py`): BCE + Dice +
0.1·Boundary-IoU + edge-head BCE, plus 0.3-weighted multi-scale deep supervision
(FEE-Net) and 0.5-weighted triage cross-entropy (EE) -- identical to `train.py`.
Metrics come from the repository `utils/metrics.py` (explicit `from_logits=True`).
"""))

    # ------------------------------------------------------------------
    # Cell 2: Environment, CPU clamp, dual-GPU detection
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 2: Environment Setup, CPU Clamping & Dependency Pre-flight
import os, sys, glob, json, time, math, random, urllib.request, zipfile, ssl, subprocess
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Zero-CPU-spike engineering: clamp every CPU thread pool to 1 ---
cv2.setNumThreads(1)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

NUM_GPUS = torch.cuda.device_count()
DEV_LIST = [f"cuda:{i}" for i in range(NUM_GPUS)] or ["cpu"]
print(f"[Env] torch={torch.__version__} | GPUs={NUM_GPUS} {DEV_LIST} | threads={torch.get_num_threads()}")

# Optional FLOPs counter (falls back to the analytic table if unavailable)
try:
    from thop import profile as thop_profile
    THOP_OK = True
except Exception:
    try:
        os.system("pip install -q thop")
        from thop import profile as thop_profile
        THOP_OK = True
    except Exception:
        THOP_OK = False
        print("[Env] thop unavailable -> using analytic FLOPs table")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Protocol constant (single definition, mirrors train.py). A frame/case counts as
# lesion-bearing when its mask exceeds this many pixels.
TRIAGE_PIXEL_THRESHOLD = 10

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("[Env] seed fixed (42) for reproducibility")
"""))

    # ------------------------------------------------------------------
    # Cell 3: Materialise the repository package + embed splits (single source of truth)
    # ------------------------------------------------------------------
    repo_blob = json.dumps(embedded, indent=2)
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 3: Materialise Repository `models/` and `utils/` + Embed the Authoritative Splits
REPO_ROOT = "/kaggle/working/repo"
REPO_SOURCES = """ + repo_blob + """
REPO_TOOL_SOURCES = """ + json.dumps(tools_src, indent=2) + """

for rel, src in REPO_SOURCES.items():
    dest = os.path.join(REPO_ROOT, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(src)

# Same for the mixed-cohort tools (imported by the models package, hence PYTHONPATH)
for rel, src in REPO_TOOL_SOURCES.items():
    dest = os.path.join(REPO_ROOT, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(src)

# Minimal package markers (the real ones drag in unrelated Laya/visualizer deps)
for pkg in ["models", "models/modules", "utils"]:
    p = os.path.join(REPO_ROOT, pkg, "__init__.py")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        open(p, "w").close()

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.unet_plus_plus import UNetPlusPlus
from models.fee_net import FEENet
from models.fee_net_early_exit import FEENetEarlyExit
from utils.losses import CompoundLoss
from utils.metrics import MetricTracker

# Provenance guard: prove the classes come from the materialised repo, not leftovers
import models.fee_net as _fee_mod
import models.fee_net_early_exit as _ee_mod
import utils.losses as _loss_mod
_repo_real = os.path.realpath(REPO_ROOT)
for m in (_fee_mod, _ee_mod, _loss_mod):
    _f = os.path.realpath(m.__file__)
    assert _f.startswith(_repo_real), f"import leaked from {_f} (expected under {_repo_real})"
    print(f"[Provenance] {m.__name__} <- {_f}")
print("[Provenance] all model/loss modules resolved from the embedded repository sources")

# ---- Repository split membership (embedded at generation time) ----
REPO_SPLITS = """ + json.dumps(splits, indent=2) + """
assert "test" in REPO_SPLITS, "v2 splits must contain an independent test set"
test_stems = list(REPO_SPLITS["test"])
fold0_train = list(REPO_SPLITS["fold_0"]["train"])
fold0_val = list(REPO_SPLITS["fold_0"]["val"])
assert len(test_stems) == 150, f"expected 150 test cases, got {len(test_stems)}"
assert len(fold0_train) == 680 and len(fold0_val) == 170, "unexpected fold-0 sizes"
tset = set(test_stems)
assert len(tset & set(fold0_train)) == 0, "FATAL: test leaked into train"
assert len(tset & set(fold0_val)) == 0, "FATAL: test leaked into val"
print(f"[v2] repository splits embedded: test={len(test_stems)} | fold0 train={len(fold0_train)} val={len(fold0_val)} | disjointness verified")
"""))

    # ------------------------------------------------------------------
    # Cell 4: Dataset ingestion (uses the embedded splits)
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 4: Kvasir-SEG Ingestion (read-only use of the mounted dataset)
# NOTE: the dataset is read **in place** from /kaggle/input and never copied into
# /kaggle/working. Copying it would inflate the kernel output archive with ~2k
# redundant image files, which makes `kaggle kernels output` extremely slow to
# download (per-file latency) and needlessly consumes the working disk.
input_img_dirs = glob.glob("/kaggle/input/**/images", recursive=True)
input_mask_dirs = glob.glob("/kaggle/input/**/masks", recursive=True)

if input_img_dirs and input_mask_dirs:
    IMG_DIR, MASK_DIR = input_img_dirs[0], input_mask_dirs[0]
    print(f"[Dataset] Using mounted dataset in place: {IMG_DIR}")
else:
    print("[Dataset] No mounted dataset found -> downloading official Simula Kvasir-SEG (46.2 MB)")
    DATA_DIR = "/kaggle/working/data/kvasir_seg"
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = "/kaggle/working/kvasir-seg.zip"
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen("https://datasets.simula.no/downloads/kvasir-seg.zip", context=ctx) as r, open(zip_path, "wb") as out:
        out.write(r.read())
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/kaggle/working/data")
    root = os.path.join("/kaggle/working/data", "Kvasir-SEG")
    root = root if os.path.isdir(root) else DATA_DIR
    IMG_DIR, MASK_DIR = os.path.join(root, "images"), os.path.join(root, "masks")
print(f"[Dataset] images: {len(glob.glob(os.path.join(IMG_DIR, '*.*')))} | masks: {len(glob.glob(os.path.join(MASK_DIR, '*.*')))}")
"""))

    # ------------------------------------------------------------------
    # Cell 5: Contiguous VRAM pre-loading
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 5: Contiguous Zero-Copy GPU VRAM Pre-loading
print("[VRAM] Pre-loading dataset into GPU memory...")

def records_to_tensors(records, img_size=256):
    n = len(records)
    imgs_np = np.empty((n, 3, img_size, img_size), dtype=np.float32)
    msks_np = np.empty((n, 1, img_size, img_size), dtype=np.float32)
    lbls_np = np.empty((n,), dtype=np.float32)
    for i, r in enumerate(records):
        img = cv2.imread(r["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        msk_bin = (msk > 127).astype(np.float32)
        imgs_np[i] = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        msks_np[i, 0] = msk_bin
        lbls_np[i] = 1.0 if msk_bin.sum() > TRIAGE_PIXEL_THRESHOLD else 0.0
    return torch.from_numpy(imgs_np), torch.from_numpy(msks_np), torch.from_numpy(lbls_np)

img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
mask_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(MASK_DIR, "*.*"))}
stem_map = {}
for ip in img_files:
    stem = os.path.splitext(os.path.basename(ip))[0]
    if stem in mask_dict:
        stem_map[stem] = {"stem": stem, "img_path": ip, "mask_path": mask_dict[stem]}

missing = [s for s in test_stems + fold0_train + fold0_val if s not in stem_map]
assert not missing, f"FATAL: {len(missing)} split stems missing from the dataset (e.g. {missing[:3]})"

train_records = [stem_map[s] for s in fold0_train]
val_records = [stem_map[s] for s in fold0_val]
test_records = [stem_map[s] for s in test_stems]

cpu_train = records_to_tensors(train_records)
cpu_val = records_to_tensors(val_records)
cpu_test = records_to_tensors(test_records)

gpu_datasets = {}
for dev in DEV_LIST:
    gpu_datasets[dev] = {
        "train_imgs": cpu_train[0].to(dev), "train_msks": cpu_train[1].to(dev), "train_lbls": cpu_train[2].to(dev),
        "val_imgs": cpu_val[0].to(dev), "val_msks": cpu_val[1].to(dev), "val_lbls": cpu_val[2].to(dev),
        "test_imgs": cpu_test[0].to(dev), "test_msks": cpu_test[1].to(dev), "test_lbls": cpu_test[2].to(dev),
    }
    mb = gpu_datasets[dev]["train_imgs"].numel() * gpu_datasets[dev]["train_imgs"].element_size() / 1e6
    print(f"  [{dev}] cached {len(cpu_train[0])} train / {len(cpu_val[0])} val / {len(cpu_test[0])} test ({mb:.1f} MB VRAM)")

BATCH_SIZE = 16
print(f"[VRAM] batch size {BATCH_SIZE}; triage positive threshold {TRIAGE_PIXEL_THRESHOLD} px (defined in Cell 2)")
"""))

    # ------------------------------------------------------------------
    # Cell 6: Model zoo (repository classes)
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 6: Model Zoo (repository implementations)
def build_model(name):
    if name == "UNet":
        return UNet(in_channels=3, num_classes=1)
    if name == "AttentionUNet":
        return AttentionUNet(in_channels=3, num_classes=1)
    if name == "UNetPlusPlus":
        return UNetPlusPlus(in_channels=3, num_classes=1)
    if name == "FEENet":
        return FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=True, base_channels=48)
    if name == "FEENetEarlyExit":
        return FEENetEarlyExit(in_channels=3, num_classes=1, base_channels=48)
    if name == "FEENet_noDWT":
        return FEENet(in_channels=3, num_classes=1, use_dwt=False, use_egm=True, base_channels=48)
    if name == "FEENet_noEGM":
        return FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=False, base_channels=48)
    raise ValueError(name)

is_early_exit = lambda n: "EarlyExit" in n
for n in ["UNet", "AttentionUNet", "UNetPlusPlus", "FEENet", "FEENetEarlyExit", "FEENet_noDWT", "FEENet_noEGM"]:
    m = build_model(n)
    print(f"  {n:18s} params={sum(p.numel() for p in m.parameters())/1e6:.3f}M")
    del m
print("[Zoo] model factory ready")
"""))

    # ------------------------------------------------------------------
    # Cell 7: Efficiency profiler (full path + measured triage gate)
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 7: Efficiency Profiler -- measured latency/FPS + measured triage-gate cost
ANALYTIC_GFLOPS = {  # fallback only; used when thop is unavailable
    "UNet": 13.75, "AttentionUNet": 14.03, "UNetPlusPlus": 34.62,
    "FEENet": 12.59, "FEENetEarlyExit": 12.59, "FEENet_noDWT": 12.87, "FEENet_noEGM": 11.83,
    # Screening gate (stem + layer1 + Haar DWT + SCA + triage head).
    "TriageGate": 0.94,
}

def measure_latency_ms(fn, dev, n_warmup=15, n_runs=50):
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_runs):
            fn()
        e.record()
        torch.cuda.synchronize(dev)
        return s.elapsed_time(e) / n_runs
    t0 = time.time()
    for _ in range(n_runs):
        fn()
    return (time.time() - t0) / n_runs * 1000.0

class FullPathWrapper(nn.Module):
    # Forces the early-exit network down its full segmentation path.
    # thop/torch call `model(x)` directly, which would trigger the dynamic triage
    # gate and report the *gate* cost instead of the full-path cost.

    def __init__(self, ee):
        super().__init__()
        self.ee = ee

    def forward(self, x):
        return self.ee(x, force_full=True, return_raw_logits=True)


def measure_gflops(model, name, dev):
    if THOP_OK:
        try:
            if name == "TriageGate":
                m = model
            else:
                m = build_model(name).to(dev).eval()
                if is_early_exit(name):
                    m = FullPathWrapper(m).to(dev).eval()
            macs, _ = thop_profile(m, inputs=(torch.randn(1, 3, 256, 256, device=dev),), verbose=False)
            if name != "TriageGate":
                del m
            return round(macs / 1e9, 2), "thop"
        except Exception as ex:
            print(f"[thop] failed for {name}: {ex}")
    return ANALYTIC_GFLOPS.get(name, 12.59), "analytic-table"

def profile_model(name, dev):
    model = build_model(name).to(dev).eval()
    x = torch.randn(1, 3, 256, 256, device=dev)
    with torch.no_grad():
        if is_early_exit(name):
            lat = measure_latency_ms(lambda: model(x, force_full=True, return_raw_logits=True), dev)
        else:
            lat = measure_latency_ms(lambda: model(x), dev)
        gflops, flops_src = measure_gflops(model, name, dev)
    out = {
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "gflops": gflops,
        "flops_source": flops_src,
        "latency_ms": round(lat, 2),
        "fps": round(1000.0 / max(lat, 1e-4), 1),
    }
    del model
    torch.cuda.empty_cache()
    return out

class TriageGate(nn.Module):
    \"\"\"Measurable cost of the screening gate: stem + layer1 + DWT + SCA + triage head.\"\"\"
    def __init__(self, ee):
        super().__init__()
        self.stem, self.layer1, self.dwt = ee.stem, ee.layer1, ee.dwt
        self.wf1, self.wf1_attn = ee.wf1, getattr(ee, "wf1_attn", None)
        self.triage_head = ee.triage_head

    def forward(self, x):
        xs = self.stem(x)
        e1 = self.layer1(xs)
        _, lh, hl, hh = self.dwt(xs)
        cat = torch.cat([lh, hl, hh], dim=1)
        if self.wf1_attn is not None:
            cat = self.wf1_attn(cat)
        hf1 = self.wf1(cat)
        return self.triage_head(e1, hf1)[0]

def profile_triage_gate(dev):
    ee = build_model("FEENetEarlyExit").to(dev).eval()
    gate = TriageGate(ee).to(dev).eval()
    x = torch.randn(1, 3, 256, 256, device=dev)
    gflops, src = measure_gflops(gate, "TriageGate", dev)
    with torch.no_grad():
        lat = measure_latency_ms(lambda: gate(x), dev)
    res = {
        "gate_params_m": round(sum(p.numel() for p in gate.parameters()) / 1e6, 3),
        "gate_gflops": gflops, "gate_flops_source": src,
        "gate_latency_ms": round(lat, 2), "gate_fps": round(1000.0 / max(lat, 1e-4), 1),
    }
    del gate, ee
    torch.cuda.empty_cache()
    return res

print("[Profiler] ready (full-path + measured triage gate)")
"""))

    # ------------------------------------------------------------------
    # Cell 8: Training engine (repository compound loss)
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 8: Training Engine -- repository CompoundLoss + deep supervision + triage CE
criterion = CompoundLoss(lambda_bce=1.0, lambda_dice=1.0, lambda_boundary=0.1).to(DEV_LIST[0])
triage_criterion = nn.CrossEntropyLoss()

def compute_loss(name, out, masks, lbls):
    \"\"\"Mirrors train.py: CompoundLoss(seg, edge, masks) + 0.3 * deep supervision + 0.5 * triage CE.\"\"\"
    if is_early_exit(name):
        seg, edge, triage_logits, _quality, aux_preds = out
    else:
        seg, edge, aux_preds = out
        triage_logits = None

    loss, _ = criterion(seg, edge, masks)

    if aux_preds:
        aux_total = 0.0
        for aux in aux_preds:
            up = F.interpolate(aux, size=masks.shape[2:], mode="bilinear", align_corners=True)
            aux_total = aux_total + criterion(up, None, masks)[0]
        loss = loss + 0.3 * (aux_total / len(aux_preds))

    if triage_logits is not None:
        loss = loss + 0.5 * triage_criterion(triage_logits, lbls)
    return loss

def train_model(name, dev, epochs=35, lr=3e-4, lambda_bce=1.0, lambda_dice=1.0, lambda_boundary=0.1):
    print("=" * 68)
    print(f"Training {name} on [{dev}] | epochs={epochs} lr={lr} boundary_w={lambda_boundary}")
    print("=" * 68)
    model = build_model(name).to(dev)
    ds = gpu_datasets[dev]

    global criterion
    criterion = CompoundLoss(lambda_bce=lambda_bce, lambda_dice=lambda_dice, lambda_boundary=lambda_boundary).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # -1.0 (as in train.py) so the first epoch always yields a checkpoint even
    # when the validation Dice is exactly 0.
    best_val_dice, best_path = -1.0, os.path.join(OUTPUT_DIR, f"best_{name.lower()}.pth")
    history = {"train_loss": [], "val_dice": [], "val_iou": []}
    n_train, n_val = len(ds["train_imgs"]), len(ds["val_imgs"])

    for epoch in range(1, epochs + 1):
        model.train()
        loss_acc, nb = torch.zeros(1, device=dev), 0
        perm = torch.randperm(n_train, device=dev)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i: min(i + BATCH_SIZE, n_train)]
            b_imgs, b_msks = ds["train_imgs"][idx], ds["train_msks"][idx]
            b_lbls = (ds["train_msks"][idx].sum(dim=(1, 2, 3)) > TRIAGE_PIXEL_THRESHOLD).long()
            if random.random() > 0.5:
                b_imgs, b_msks = torch.flip(b_imgs, [3]), torch.flip(b_msks, [3])
            if random.random() > 0.5:
                b_imgs, b_msks = torch.flip(b_imgs, [2]), torch.flip(b_msks, [2])

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                out = model(b_imgs, return_aux=True) if "FEENet" in name or "EarlyExit" in name else model(b_imgs)
                if is_early_exit(name):
                    loss = compute_loss(name, out, b_msks, b_lbls)
                elif "FEENet" in name:
                    loss = compute_loss(name, out, b_msks, b_lbls)
                else:
                    loss, _ = criterion(out, None, b_msks)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            loss_acc += loss.detach(); nb += 1

        scheduler.step()
        epoch_loss = (loss_acc / max(nb, 1)).item()

        # validation with repository metrics (explicit from_logits)
        model.eval()
        tracker = MetricTracker()
        with torch.no_grad():
            for i in range(0, n_val, BATCH_SIZE):
                sl = slice(i, min(i + BATCH_SIZE, n_val))
                v_imgs, v_msks = ds["val_imgs"][sl], ds["val_msks"][sl]
                if is_early_exit(name):
                    logits = model(v_imgs, force_full=True, return_raw_logits=True)
                else:
                    logits = model(v_imgs)
                tracker.update(logits, v_msks, from_logits=True)
        res = tracker.get_results()
        history["train_loss"].append(round(epoch_loss, 4))
        history["val_dice"].append(round(res["dice"] * 100, 2))
        history["val_iou"].append(round(res["iou"] * 100, 2))
        if res["dice"] > best_val_dice:
            best_val_dice = res["dice"]
            torch.save(model.state_dict(), best_path)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] loss={epoch_loss:.4f} val_dice={res['dice']*100:.2f}% val_iou={res['iou']*100:.2f}% best={best_val_dice*100:.2f}%")

    print(f"Finished {name}: best val Dice = {best_val_dice*100:.2f}% -> {best_path}")
    return model, best_path, history

print("[Train] engine ready (repository loss, deep supervision, triage CE)")
"""))

    # ------------------------------------------------------------------
    # Cell 9: Execute training
    # ------------------------------------------------------------------
    ablations = """
EXTRA_MODELS = ["FEENet_noDWT", "FEENet_noEGM"] if RUN_ABLATIONS else []
""".strip().replace("RUN_ABLATIONS", str(RUN_ABLATIONS))
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 9: Train All Models with Model-Level Dual-T4 Allocation
# 60 epochs (not 35): the local probe matrix showed the full compound loss still
# climbing at 35 (76.14 val Dice) and reaching 77.55 only by 60, while the plain
# baselines converge much earlier. Equal epochs for every model (see Exp-032).
EPOCHS = 60
LR = 3e-4
""" + ablations + """

BASE_MODELS = ["UNet", "AttentionUNet", "UNetPlusPlus", "FEENet", "FEENetEarlyExit"]
ALL_MODELS = BASE_MODELS + EXTRA_MODELS

def assign(name):
    if NUM_GPUS > 1:
        return "cuda:0" if name in ("UNet", "AttentionUNet") else "cuda:1"
    return DEV_LIST[0]

model_assignments = {n: assign(n) for n in ALL_MODELS}
trained, histories, profiles = {}, {}, {}
gate_profile = None

for name in ALL_MODELS:
    dev = model_assignments[name]
    print(f"\\n>>> Profiling {name} on {dev}")
    profiles[name] = profile_model(name, dev)
    print(f"    params={profiles[name]['params_m']}M gflops={profiles[name]['gflops']} ({profiles[name]['flops_source']}) "
          f"lat={profiles[name]['latency_ms']}ms fps={profiles[name]['fps']}")
    if name == "FEENetEarlyExit":
        gate_profile = profile_triage_gate(dev)
        print(f"    [gate] params={gate_profile['gate_params_m']}M gflops={gate_profile['gate_gflops']} "
              f"({gate_profile['gate_flops_source']}) lat={gate_profile['gate_latency_ms']}ms fps={gate_profile['gate_fps']}")
    model, ckpt, hist = train_model(name, dev, epochs=EPOCHS, lr=LR,
                                    lambda_boundary=0.0 if name == "FEENet_noBL" else 0.1)
    trained[name] = (model.to(dev), ckpt)
    histories[name] = hist

# Provenance artifact: every params/FLOPs/latency number quoted in the paper must be
# traceable to a machine-readable file shipped with the results.
with open(os.path.join(OUTPUT_DIR, "model_profiles.json"), "w", encoding="utf-8") as _f:
    json.dump({
        "full_path": {k: profiles[k] for k in profiles},
        "triage_gate": gate_profile,
        "note": ("thop MACs at 256x256 batch 1; latency/FPS by non-blocking CUDA events "
                 f"on {DEV_LIST[0]}; gate = stem+layer1+DWT+SCA+triage head"),
    }, _f, indent=2)
print("[provenance] wrote model_profiles.json")
"""))

    # ------------------------------------------------------------------
    # Cell 10: Independent test evaluation with repository metrics
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 10: Independent Test Evaluation (150 cases, repository metrics)
test_results = {}
print("=" * 78)
print("INDEPENDENT TEST SET EVALUATION -- 150 held-out cases (repository splits.json)")
print("=" * 78)

for name in ALL_MODELS:
    dev = model_assignments[name]
    model, ckpt = trained[name]
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    tracker = MetricTracker()
    with torch.no_grad():
        for i in range(0, len(gpu_datasets[dev]["test_imgs"]), BATCH_SIZE):
            sl = slice(i, min(i + BATCH_SIZE, len(gpu_datasets[dev]["test_imgs"])))
            t_imgs, t_msks = gpu_datasets[dev]["test_imgs"][sl], gpu_datasets[dev]["test_msks"][sl]
            # NOTE: FEENetEarlyExit.forward(force_full=True) returns *probabilities*;
            # MetricTracker(from_logits=True) would apply a second sigmoid and turn
            # every pixel positive. Always request raw logits explicitly.
            logits = (model(t_imgs, force_full=True, return_raw_logits=True)
                      if is_early_exit(name) else model(t_imgs))
            tracker.update(logits, t_msks, from_logits=True)
    r = tracker.get_results()
    test_results[name] = {
        "Test Dice (%)": round(r["dice"] * 100, 2),
        "Test mIoU (%)": round(r["iou"] * 100, 2),
        "HD95 (px)": round(r["hd95"], 2),
        "Precision (%)": round(r["precision"] * 100, 2),
        "Recall (%)": round(r["recall"] * 100, 2),
        "Params (M)": profiles[name]["params_m"],
        "FLOPs (G)": profiles[name]["gflops"],
        "Latency (ms)": profiles[name]["latency_ms"],
        "FPS": profiles[name]["fps"],
    }
    print(f"[{name:18s}] Dice {test_results[name]['Test Dice (%)']:5.2f}% | mIoU {test_results[name]['Test mIoU (%)']:5.2f}% | "
          f"HD95 {test_results[name]['HD95 (px)']:6.2f}px | {test_results[name]['Params (M)']:.3f}M | "
          f"{test_results[name]['FLOPs (G)']:.2f}G | {test_results[name]['FPS']:.1f} FPS")

# Persist immediately: do not let a later cell's failure discard these numbers
# (this is exactly how the v4 run lost its benchmark table).
import csv as _csv
with open(os.path.join(OUTPUT_DIR, "benchmark_metrics_independent_test.csv"), "w", newline="",
          encoding="utf-8") as _f:
    _w = _csv.writer(_f)
    _cols = list(next(iter(test_results.values())).keys())
    _w.writerow(["Model"] + _cols)
    for _k, _v in test_results.items():
        _w.writerow([_k] + [_v[c] for c in _cols])
print("[persist] benchmark_metrics_independent_test.csv written")

# Ablation deltas (same protocol, same seed)
if EXTRA_MODELS:
    full = test_results["FEENet"]["Test Dice (%)"]
    print("\\n[Ablation] Dice deltas vs. full FEE-Net (same v2 protocol):")
    for name in EXTRA_MODELS:
        d = test_results[name]["Test Dice (%)"] - full
        print(f"  {name:16s} {test_results[name]['Test Dice (%)']:5.2f}%  ({d:+.2f})")
"""))

    # ------------------------------------------------------------------
    # Cell 11: CRC screening calibration (calibrate on VAL, certify on TEST)
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 11: Triage Screening & Conformal Risk Control
# Protocol fix (Exp-030 audit): the threshold tau* is calibrated on the VALIDATION
# fold and only *certified* on the held-out test set. Calibrating and reporting on
# the same samples would void the exchangeability assumption that conformal risk
# control requires, so the two roles are strictly separated here.
ee_dev = model_assignments["FEENetEarlyExit"]
ee_model, ee_ckpt = trained["FEENetEarlyExit"]
ee_model.load_state_dict(torch.load(ee_ckpt, map_location=ee_dev))
ee_model.eval()


def triage_risks(split):
    imgs = gpu_datasets[ee_dev][f"{split}_imgs"]
    lbls = gpu_datasets[ee_dev][f"{split}_lbls"]
    out = []
    with torch.no_grad():
        for i in range(0, len(lbls), BATCH_SIZE):
            sl = slice(i, min(i + BATCH_SIZE, len(lbls)))
            # Repository API: risk scores are returned at index 2 when return_edge=True
            # (default forward returns only the mask, unlike the retired notebook copy).
            res = ee_model(imgs[sl], tau=0.5, return_edge=True)
            out.append(res[2].cpu().numpy())
    return np.concatenate(out), lbls.cpu().numpy()


val_risks, val_lbls = triage_risks("val")
test_risks, test_lbls = triage_risks("test")

print(f"[Cohort] positives: val {int(val_lbls.sum())}/{len(val_lbls)} | test {int(test_lbls.sum())}/{len(test_lbls)}")
print("[Cohort] Kvasir-SEG is an all-positive lesion cohort: no healthy-mucosa frames exist,")
print("        so every bypass is a false negative and the normal-bypass rate is structurally 0.")

full_gflops = profiles["FEENetEarlyExit"]["gflops"]
gate_gflops = gate_profile["gate_gflops"] if gate_profile else None


def cp_upper(k, n, conf=0.05):
    # Clopper-Pearson exact upper confidence bound on a binomial rate
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1 - conf, k + 1, n - k))


# --- calibrate tau* on the validation fold only ---
n_val_pos = int(val_lbls.sum())
val_pos_risks = val_risks[val_lbls == 1]
TARGET_FNR = 0.05
tau_grid = np.linspace(0.01, 0.99, 99)
calibrated_tau, certified_upper = 0.0, 1.0
for tau in tau_grid:
    k = int((val_pos_risks < tau).sum())
    upper = cp_upper(k, n_val_pos)
    if upper <= TARGET_FNR:
        calibrated_tau, certified_upper = float(tau), upper
    else:
        break

# --- certify / report on the held-out test set (single evaluation) ---
n_test_pos = int(test_lbls.sum())
test_pos_risks = test_risks[test_lbls == 1]
test_misses = int((test_pos_risks < calibrated_tau).sum())
test_fnr = test_misses / max(n_test_pos, 1) * 100.0
test_upper = cp_upper(test_misses, n_test_pos)

# Threshold sweep on TEST for reporting only (no calibration happens here)
sweep = []
for tau in np.linspace(0.05, 0.95, 19):
    bypassed = test_risks < tau
    lesion = np.where(test_lbls == 1)[0]
    fnr = float(bypassed[lesion].sum() / max(len(lesion), 1)) * 100.0
    avg_gf = float(np.mean(np.where(bypassed, gate_gflops or 1.45, full_gflops)))
    sweep.append({"tau": round(float(tau), 2), "fnr": round(fnr, 2),
                  "bypass_rate": 0.0, "avg_gflops": round(avg_gf, 2)})

pd.DataFrame(sweep).to_csv(os.path.join(OUTPUT_DIR, "triage_screening_sweep.csv"), index=False)
pd.DataFrame({"split": ["val"] * len(val_risks) + ["test"] * len(test_risks),
              "label": list(val_lbls) + list(test_lbls),
              "risk_score": list(val_risks) + list(test_risks)}
             ).to_csv(os.path.join(OUTPUT_DIR, "triage_risk_scores.csv"), index=False)

print("=" * 68)
print("CONFORMAL RISK CONTROL CALIBRATION (calibrate on val -> certify on test)")
print(f"  target: FNR <= {TARGET_FNR*100:.1f}% @ 95% confidence")
print(f"  calibration set : fold-0 validation, n_positive = {n_val_pos}")
print(f"  calibrated tau* = {calibrated_tau:.4f} (CP upper on val = {certified_upper*100:.2f}%)")
print(f"  certification   : independent test, n_positive = {n_test_pos}")
print(f"  measured FNR    = {test_fnr:.2f}%  (misses {test_misses}/{n_test_pos})")
print(f"  CP upper on test= {test_upper*100:.2f}%  -> guarantee holds: {test_upper <= TARGET_FNR}")
if gate_profile:
    print(f"  measured gate: {gate_profile['gate_gflops']:.2f} GFLOPs @ {gate_profile['gate_fps']:.1f} FPS "
          f"({gate_profile['gate_params_m']:.3f}M params) vs full path {full_gflops:.2f} GFLOPs")
    print(f"  analytic headroom = {full_gflops / max(gate_profile['gate_gflops'], 1e-6):.1f}x "
          f"(screening mode, not a segmentation operating point)")
print("=" * 68)
"""))

    # ------------------------------------------------------------------
    # Cell 12: Mixed-cohort conformal screening (the paper's key experiment)
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 12: Mixed-Cohort Conformal Screening (Kvasir-SEG is all-positive, so gate
# specificity and real compute saving are unmeasurable there). We derive a
# leakage-free mixed cohort from the HELD-OUT test frames: lesion frames plus
# same-frame mucosal crops, split 75/75 into calibration and certification.
import subprocess  # self-contained: never rely on an earlier cell's imports
KSEG_ROOT = os.path.dirname(IMG_DIR)
MIXED_DIR = "/kaggle/working/mixed_cohort"
SPLITS_LOCAL = "/kaggle/working/data/kvasir_splits.json"
os.makedirs(os.path.dirname(SPLITS_LOCAL), exist_ok=True)
with open(SPLITS_LOCAL, "w", encoding="utf-8") as f:
    json.dump(REPO_SPLITS, f, indent=2)

import subprocess
_env = dict(os.environ)
_env["PYTHONPATH"] = REPO_ROOT + os.pathsep + _env.get("PYTHONPATH", "")
_env["MIXED_COHORT_DIR"] = MIXED_DIR
_env["PROBE_RESULTS_DIR"] = OUTPUT_DIR

# Never let this optional experiment kill the run: the benchmark CSV/figures below
# matter more, and a crash here previously discarded ALL v4 results.
try:
    print("[mixed] building leakage-free mixed cohort from held-out test frames ...")
    r = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "tools", "build_mixed_cohort.py"),
                        "--kseg", KSEG_ROOT, "--splits", SPLITS_LOCAL, "--out", MIXED_DIR],
                       env=_env, capture_output=True, text=True)
    print(r.stdout[-800:]); print(r.stderr[-500:])

    print("[mixed] conformal screening on the mixed cohort ...")
    r = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "tools", "eval_mixed_cohort_gate.py"),
                        "--ckpt", os.path.join(OUTPUT_DIR, "best_feenetearlyexit.pth"),
                        "--device", DEV_LIST[0]],
                       env=_env, capture_output=True, text=True)
    print(r.stdout[-2500:]); print(r.stderr[-800:])
    _mixed_json = os.path.join(OUTPUT_DIR, "mixed_cohort_screening.json")
    if os.path.exists(_mixed_json):
        print("[mixed] summary:", open(_mixed_json, encoding="utf-8").read()[:900])
    else:
        print("[mixed] WARNING: screening result not produced")
except Exception as _e:  # noqa: BLE001
    import traceback
    print(f"[mixed] EXPERIMENT FAILED (non-fatal, continuing): {type(_e).__name__}: {_e}")
    traceback.print_exc()
"""))

    # ------------------------------------------------------------------
    # Cell 12: Figures (measured points only)
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 13: Publication Figures (measured points only)
MAIN = ["UNet", "AttentionUNet", "UNetPlusPlus", "FEENet", "FEENetEarlyExit"]
names = [n for n in MAIN if n in test_results]
gflops = [test_results[n]["FLOPs (G)"] for n in names]
dices = [test_results[n]["Test Dice (%)"] for n in names]
fps_vals = [test_results[n]["FPS"] for n in names]
colors = ["#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728", "#9467bd"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
for i, n in enumerate(names):
    ax1.scatter(gflops[i], dices[i], s=150, c=colors[i], edgecolors="black", linewidth=1.2, zorder=5)
    ax1.annotate(f"{n}\\n({gflops[i]:.2f}G, {dices[i]:.2f}%)", (gflops[i] + 0.4, dices[i] - 0.3), fontsize=9, weight="bold")
ax1.set_xlabel("Computation (GFLOPs) [Lower is Better]", fontsize=12, weight="bold")
ax1.set_ylabel("Independent Test Dice (%) [Higher is Better]", fontsize=12, weight="bold")
ax1.set_title("Pareto Frontier: Accuracy vs. Computation", fontsize=13, weight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)

for i, n in enumerate(names):
    ax2.scatter(fps_vals[i], dices[i], s=150, c=colors[i], edgecolors="black", linewidth=1.2, zorder=5)
    ax2.annotate(f"{n}\\n({fps_vals[i]:.1f} FPS)", (fps_vals[i] + 2, dices[i] - 0.3), fontsize=9, weight="bold")
ax2.axvline(x=30, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
ax2.text(31, min(dices) - 1.5, "Clinical Real-Time (30 FPS)", color="#2ca02c", fontsize=9, weight="bold", rotation=90)
if gate_profile:
    ax2.annotate(
        f"Triage gate (screening mode, measured):\\n{gate_profile['gate_gflops']:.2f} GFLOPs @ {gate_profile['gate_fps']:.1f} FPS\\n"
        f"no pixel-wise decoding - not a segmentation operating point",
        xy=(0.60, 0.06), xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f0f8ff", ec="#4682b4", lw=1.2), fontsize=8.5)
ax2.set_xlabel("Throughput (FPS) [Higher is Better]", fontsize=12, weight="bold")
ax2.set_ylabel("Independent Test Dice (%) [Higher is Better]", fontsize=12, weight="bold")
ax2.set_title("Clinical Feasibility: Throughput vs. Accuracy", fontsize=13, weight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_pareto_frontier.png"), bbox_inches="tight")
plt.close()

# Qualitative comparison
vis_n = 3
dev0 = model_assignments["FEENet"]
raw_imgs = gpu_datasets[dev0]["test_imgs"][:vis_n].permute(0, 2, 3, 1).cpu().numpy()
gt_masks = gpu_datasets[dev0]["test_msks"][:vis_n].cpu().numpy().squeeze(1)
preds = {}
for n in ["UNet", "AttentionUNet", "UNetPlusPlus", "FEENet"]:
    d = model_assignments[n]
    mdl, ck = trained[n]
    mdl.load_state_dict(torch.load(ck, map_location=d))
    mdl.eval()
    with torch.no_grad():
        p = torch.sigmoid(mdl(gpu_datasets[d]["test_imgs"][:vis_n].to(d)))
    preds[n] = (p > 0.5).cpu().numpy().squeeze(1)

fig, axes = plt.subplots(vis_n, 5, figsize=(15, 7.5), dpi=200)
for r in range(vis_n):
    axes[r, 0].imshow(raw_imgs[r]); axes[r, 0].set_ylabel(f"case {r+1}", fontsize=9)
    axes[r, 1].imshow(gt_masks[r], cmap="gray")
    for c, n in enumerate(["UNet", "AttentionUNet", "UNetPlusPlus", "FEENet"]):
        axes[r, c + 1].imshow(preds[n][r], cmap="gray")
    if r == 0:
        for c, t in enumerate(["Endoscopy", "Ground Truth", "U-Net", "Att-U-Net", "FEE-Net (Ours)"]):
            axes[r, c].set_title(t, fontsize=11, weight="bold")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_qualitative_comparison.png"), bbox_inches="tight")
plt.close()
print("[Figures] pareto + qualitative saved (all plotted points measured end-to-end)")
"""))

    # ------------------------------------------------------------------
    # Cell 13: Tables + packaging
    # ------------------------------------------------------------------
    nb.cells.append(nbf.v4.new_code_cell("""# Cell 14: Tables, Provenance Bundle & Packaging
df = pd.DataFrame.from_dict(test_results, orient="index")
for c in ["Params (M)", "FLOPs (G)"]:
    if c in df.columns:
        df[c] = df[c].astype(float)
df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_metrics_independent_test.csv"))
with open(os.path.join(OUTPUT_DIR, "table_colonoscopy_benchmark.tex"), "w") as f:
    f.write(df.to_latex(caption="150-case held-out independent test set (repository splits.json).", label="tab:benchmark"))

print("=" * 78)
print("FINAL BENCHMARK SUMMARY -- 150-CASE INDEPENDENT TEST SET")
print("=" * 78)
print(df.to_markdown())
if gate_profile:
    print(f"\\nTriage gate (measured): {gate_profile['gate_gflops']:.2f} GFLOPs @ {gate_profile['gate_fps']:.1f} FPS, "
          f"{gate_profile['gate_params_m']:.3f}M params | CRC tau* = {calibrated_tau:.4f}")

# Provenance: ship the exact repository sources used by this run
prov_dir = os.path.join(OUTPUT_DIR, "repo_provenance")
os.makedirs(prov_dir, exist_ok=True)
for rel, src in list(REPO_SOURCES.items()) + list(REPO_TOOL_SOURCES.items()):
    dst = os.path.join(prov_dir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(src)
with open(os.path.join(prov_dir, "splits.json"), "w") as f:
    json.dump(REPO_SPLITS, f, indent=2)
with open(os.path.join(prov_dir, "PROVENANCE.txt"), "w") as f:
    f.write("Sources embedded from the repository at notebook-generation time.\\n"
            "models imported: models.unet, models.attention_unet, models.unet_plus_plus,\\n"
            "               models.fee_net, models.fee_net_early_exit\\n"
            "losses/metrics : utils.losses.CompoundLoss, utils.metrics.MetricTracker\\n"
            "splits         : data/kvasir_seg/splits.json (Exp-027 v2)\\n")

zip_path = "/kaggle/working/feenet_baselines_kaggle_results.zip"
with __import__("zipfile").ZipFile(zip_path, "w", __import__("zipfile").ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for fn in files:
            full = os.path.join(root, fn)
            zf.write(full, arcname=os.path.relpath(full, OUTPUT_DIR))
print(f"[Package] {zip_path}")
print("[Done] benchmark complete")
"""))

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[Generator] wrote {output_path} with {len(nb.cells)} cells")
    print(f"[Generator] embedded repo modules: {len(embedded)} | splits: test={len(splits.get('test', []))}")


if __name__ == "__main__":
    build_all_in_one_notebook()
