"""
FPGA Fault Diagnosis -- Hyperparameter Tuning Kernel

Runs a small grid of FPGA-friendly Transformer configs on Kaggle T4.

Protocol:
- DataLoaders are built ONCE; all configs share the same in-memory TensorDataset.
- Validation set only is used to rank configs. Test set is NEVER seen during selection.
- Test evaluation runs exactly ONCE for the winner.
- Results are written to JSON after every config so partial runs are recoverable.
- A global time budget prevents exceeding the 9-hour Kaggle session limit.

Official baseline (run 20260615_223905_fpga_fault_final_t4):
  test_acc = 98.22%  |  fault_recall = 99.75%  |  checkpoint = 381 KB
"""

import copy
import datetime
import glob
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -- Session clock --------------------------------------------------------------
SESSION_START = time.perf_counter()
MAX_BUDGET_SECONDS = 8 * 3600     # 8-hour cap; Kaggle limit is 9 h
MIN_PER_CONFIG_SECONDS = 80 * 60  # skip a config if < 80 min remain


def log(msg):
    elapsed = (time.perf_counter() - SESSION_START) / 60
    print(f"[{elapsed:6.1f}m] {msg}", flush=True)


def elapsed_seconds():
    return time.perf_counter() - SESSION_START


def remaining_seconds():
    return MAX_BUDGET_SECONDS - elapsed_seconds()


# -- Kaggle path helpers --------------------------------------------------------
WORKING_DIR = "/kaggle/working"
INPUT_ROOT  = "/kaggle/input"


def locate_source_root(input_root):
    hits = glob.glob(
        os.path.join(input_root, "**", "src", "config.py"), recursive=True
    )
    if not hits:
        raise FileNotFoundError(
            "Cannot find src/config.py -- attach mehrdadsheikhabbasi/fpga-fault-src."
        )
    return os.path.dirname(os.path.dirname(hits[0]))


def detect_mafaulda_root(input_root):
    markers = (
        "normal", "imbalance",
        "horizontal-misalignment", "vertical-misalignment",
        "overhang", "underhang",
    )
    best = None
    for dirpath, dirnames, _ in os.walk(input_root):
        lowered = {d.lower() for d in dirnames}
        hits = sum(1 for m in markers if m in lowered)
        if hits >= 3:
            if best is None or hits > best[0]:
                best = (hits, dirpath)
    return best[1] if best else None


# -- Tuning grid ----------------------------------------------------------------
#
# Configs are ordered from smallest -> largest to maximise coverage within
# the time budget. Configs that cannot start within MIN_PER_CONFIG_SECONDS
# are skipped but noted in the final report.
#
# Extra config added beyond the user's A-E:
#   F_1layer -- single encoder layer. If 1 layer gives equivalent accuracy, the
#   FPGA encoder block count is halved, materially simplifying Stage 8 hardware.
#   This is the most FPGA-relevant ablation not in the original grid.

TUNING_CONFIGS = [
    # -- F: single-layer -- most FPGA-friendly; tests whether 2 layers are needed --
    {
        "name": "F_1layer",
        "d_model": 32, "nhead": 4, "num_layers": 1, "dim_feedforward": 64,
        "dropout": 0.1, "lr": 1e-3,
        "note": "1 encoder layer; halves FPGA encoder HW; check if depth matters",
    },
    # -- A: exact baseline architecture -- within-grid reproducibility reference --
    {
        "name": "A_baseline",
        "d_model": 32, "nhead": 4, "num_layers": 2, "dim_feedforward": 64,
        "dropout": 0.1, "lr": 1e-3,
        "note": "Official baseline arch; validates tuning run vs 98.22% target",
    },
    # -- B: wider FFN, same embedding -- cheap capacity with minimal HW delta -----
    {
        "name": "B_wide_ffn",
        "d_model": 32, "nhead": 4, "num_layers": 2, "dim_feedforward": 128,
        "dropout": 0.1, "lr": 1e-3,
        "note": "FFN 64->128; same d_model; ~1.3x weight increase; still FPGA-ok",
    },
    # -- E: 3 encoder layers -- tests depth vs baseline -------------------------
    {
        "name": "E_3layers",
        "d_model": 32, "nhead": 4, "num_layers": 3, "dim_feedforward": 64,
        "dropout": 0.1, "lr": 1e-3,
        "note": "Depth 2->3; moderate HW cost; useful if attention diversity helps",
    },
    # -- C: d_model=48 -- wider embedding; 48 is not a power of 2 (less clean) --
    {
        "name": "C_d48",
        "d_model": 48, "nhead": 4, "num_layers": 2, "dim_feedforward": 96,
        "dropout": 0.1, "lr": 1e-3,
        "note": "d_model=48 (non-pow2); FPGA fixed-point layout less clean",
    },
    # -- D: d_model=64, lower LR -- upper-bound; most resource-intensive ---------
    {
        "name": "D_d64_slow",
        "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128,
        "dropout": 0.1, "lr": 5e-4,
        "note": "Largest config; lr=5e-4; strong upper bound but heavier HW cost",
    },
]

# Fixed training hyperparameters (per user spec -- do not change)
WINDOW_SIZE  = 1024
STRIDE       = 1024
NORM_METHOD  = "zscore"
NORM_SCOPE   = "per_window"
BATCH_SIZE   = 64
MAX_EPOCHS   = 12
PATIENCE     = 3
SEED         = 42
IN_CHANNELS  = 8
NUM_CLASSES  = 4
CLASS_NAMES  = ["normal", "imbalance", "misalignment", "bearing_fault"]

# Official baseline numbers for the final comparison table
OFFICIAL_BASELINE = {
    "run_id":             "20260615_223905_fpga_fault_final_t4",
    "test_acc":           0.9822,
    "test_macro_f2":      0.9780,
    "test_weighted_f2":   0.9822,
    "fault_recall":       0.9975,
    "false_alarm_rate":   0.0234,
    "checkpoint_size_kb": 381,
    "arch": {"d_model": 32, "nhead": 4, "num_layers": 2,
             "dim_feedforward": 64, "dropout": 0.1, "lr": 1e-3},
}


# -- Reproducibility ------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -- Model (self-contained; mirrors TransformerBaseline exactly) ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + self.attn_drop(attn_out))
        x = self.ffn_norm(x + self.ffn_drop(self.ffn(x)))
        return x


class TuningTransformer(nn.Module):
    """Identical architecture to TransformerBaseline; parameterised for sweeping."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        input_dim: int = IN_CHANNELS,
        seq_len: int = WINDOW_SIZE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, seq_len, dropout)
        self.encoder    = nn.ModuleList([
            _EncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.input_proj(x))
        for layer in self.encoder:
            x = layer(x)
        return self.classifier(x.mean(dim=1))


# -- Loss -----------------------------------------------------------------------
def build_criterion(train_labels: np.ndarray, device: torch.device) -> nn.Module:
    counts = np.bincount(train_labels.astype(np.int64), minlength=NUM_CLASSES).astype(np.float32)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    total = counts.sum()
    for i, c in enumerate(counts):
        if c > 0:
            weights[i] = total / (NUM_CLASSES * c)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))


# -- Evaluation -----------------------------------------------------------------
def eval_loop(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Return (avg_loss, accuracy, confusion_matrix[NUM_CLASSES, NUM_CLASSES])."""
    model.eval()
    total_loss = total_correct = total_samples = 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            preds  = torch.argmax(logits, dim=1)
            total_loss    += loss.item() * xb.size(0)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)
            for t, p in zip(yb.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1
    return total_loss / total_samples, total_correct / total_samples, confusion


# -- Metrics from confusion matrix ----------------------------------------------
def macro_f2(cm: np.ndarray) -> float:
    f2s = []
    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = 4 * p + r
        f2s.append(5 * p * r / denom if denom > 0 else 0.0)
    return float(np.mean(f2s))


def weighted_f2(cm: np.ndarray) -> float:
    f2s, supports = [], []
    for i in range(cm.shape[0]):
        tp  = float(cm[i, i])
        fp  = float(cm[:, i].sum() - cm[i, i])
        fn  = float(cm[i, :].sum() - cm[i, i])
        p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        d   = 4 * p + r
        f2s.append(5 * p * r / d if d > 0 else 0.0)
        supports.append(int(cm[i, :].sum()))
    total = sum(supports)
    return float(np.average(f2s, weights=supports)) if total > 0 else 0.0


def safety_metrics(cm: np.ndarray) -> dict:
    """Fault recall and false-alarm rate; normal class = index 0."""
    normal_row   = cm[0, :]
    fault_rows   = cm[1:, :]
    captured     = int(fault_rows[:, 1:].sum())
    missed       = int(fault_rows[:, 0].sum())
    false_alarms = int(normal_row[1:].sum())
    actual_faults  = captured + missed
    actual_normal  = int(normal_row.sum())
    return {
        "fault_recall":      captured / actual_faults  if actual_faults  > 0 else 0.0,
        "false_alarm_rate":  false_alarms / actual_normal if actual_normal > 0 else 0.0,
        "captured_faults":   captured,
        "missed_faults":     missed,
        "false_alarms":      false_alarms,
        "actual_faults":     actual_faults,
        "actual_normal":     actual_normal,
    }


def per_class_accuracy(cm: np.ndarray) -> dict:
    return {
        CLASS_NAMES[i]: (float(cm[i, i] / cm[i, :].sum()) if cm[i, :].sum() > 0 else None)
        for i in range(cm.shape[0])
    }


# -- Single-config training -----------------------------------------------------
def train_one_config(
    cfg: dict,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tuning_dir: Path,
) -> dict:
    name = cfg["name"]
    log(f"{'=' * 60}")
    log(f"Config {name}  |  {cfg['note']}")
    log(f"  arch: d_model={cfg['d_model']} nhead={cfg['nhead']} "
        f"layers={cfg['num_layers']} ffn={cfg['dim_feedforward']} "
        f"dropout={cfg['dropout']} lr={cfg['lr']}")
    t_config_start = time.perf_counter()

    set_seed(SEED)
    model = TuningTransformer(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    ).to(device)
    n_params = count_params(model)
    log(f"  Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val_loss   = float("inf")
    best_epoch      = -1
    best_val_acc    = 0.0
    best_val_cm     = None
    best_state      = None
    no_improve      = 0
    stopped_epoch   = None

    history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")}

    for epoch in range(MAX_EPOCHS):
        t_ep = time.perf_counter()

        # -- train --
        model.train()
        total_loss = total_correct = total_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(logits, dim=1)
            total_loss    += loss.item() * xb.size(0)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)
        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # -- validate --
        val_loss, val_acc, val_cm = eval_loop(model, val_loader, criterion, device)
        ep_secs = time.perf_counter() - t_ep

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc,  6))
        history["val_loss"].append(round(val_loss,    6))
        history["val_acc"].append(round(val_acc,      6))

        log(
            f"  Ep {epoch+1:02d}/{MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"[{ep_secs:.0f}s]"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_epoch    = epoch + 1
            best_val_cm   = val_cm.copy()
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                stopped_epoch = epoch + 1
                log(f"  Early stopping at epoch {stopped_epoch}")
                break

    if stopped_epoch is None:
        stopped_epoch = history["epoch"][-1] if history["epoch"] else MAX_EPOCHS

    # -- save checkpoint --
    ckpt_dir  = tuning_dir / name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"
    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "config":           cfg,
                "epoch":            best_epoch,
                "val_loss":         best_val_loss,
                "val_acc":          best_val_acc,
            },
            ckpt_path,
        )

    config_secs = time.perf_counter() - t_config_start

    # -- compute validation metrics --
    val_macro_f2   = macro_f2(best_val_cm)   if best_val_cm is not None else None
    val_wt_f2      = weighted_f2(best_val_cm) if best_val_cm is not None else None
    val_safety     = safety_metrics(best_val_cm) if best_val_cm is not None else {}
    val_per_cls    = per_class_accuracy(best_val_cm) if best_val_cm is not None else {}

    result = {
        "name":              name,
        "config":            cfg,
        "n_params":          n_params,
        "best_epoch":        best_epoch,
        "stopped_epoch":     int(stopped_epoch),
        "best_val_loss":     float(best_val_loss),
        "val_acc":           float(best_val_acc),
        "val_macro_f2":      float(val_macro_f2)  if val_macro_f2  is not None else None,
        "val_weighted_f2":   float(val_wt_f2)     if val_wt_f2     is not None else None,
        "val_fault_recall":  float(val_safety.get("fault_recall",    0.0)),
        "val_false_alarm":   float(val_safety.get("false_alarm_rate", 0.0)),
        "val_safety":        {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                              for k, v in val_safety.items()},
        "val_per_class_acc": val_per_cls,
        "val_confusion":     best_val_cm.tolist() if best_val_cm is not None else None,
        "history":           history,
        "runtime_seconds":   round(config_secs, 1),
        "checkpoint_path":   str(ckpt_path),
    }

    # -- save per-config JSON --
    res_dir = tuning_dir / name / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    log(
        f"Config {name} done  |  "
        f"val_acc={best_val_acc:.4f}  val_macro_f2={val_macro_f2:.4f}  "
        f"best_epoch={best_epoch}  runtime={config_secs/60:.1f}m"
    )
    return result


# -- Best-config selection ------------------------------------------------------
def select_best(results: list) -> dict:
    """
    Rank by (val_macro_f2, val_acc, -val_loss).
    Using macro_f2 as primary because it balances precision/recall across
    all classes and penalises misclassification of rare classes more than
    raw accuracy does.
    """
    def _key(r):
        return (
            r.get("val_macro_f2") or 0.0,
            r.get("val_acc")      or 0.0,
            -(r.get("best_val_loss") or float("inf")),
        )
    return max(results, key=_key)


# -- FPGA-friendliness tag ------------------------------------------------------
def fpga_tag(cfg: dict) -> str:
    d   = cfg["d_model"]
    ffn = cfg["dim_feedforward"]
    nl  = cfg["num_layers"]
    pow2_d   = (d & (d - 1)) == 0
    pow2_ffn = (ffn & (ffn - 1)) == 0
    if not pow2_d:
        return "~limited"
    if nl == 1:
        return "excellent"
    if d <= 32 and ffn <= 64:
        return "excellent" if nl <= 2 else "good"
    if d <= 64 and ffn <= 128:
        return "good"
    return "moderate"


# -- Summary table --------------------------------------------------------------
def print_summary(results: list, best_name: str):
    sep = "-" * 100
    hdr = (
        f"{'Config':<15} {'Params':>7} {'BestEp':>7} {'ValLoss':>8} "
        f"{'ValAcc':>8} {'MacroF2':>8} {'WtdF2':>7} "
        f"{'FaultRec':>9} {'FalseAlm':>9} {'Runtime':>8} {'FPGA':>10}"
    )
    print(f"\n{sep}")
    print("HYPERPARAMETER TUNING  --  VALIDATION SUMMARY")
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        mark = " < BEST" if r["name"] == best_name else ""
        mf2 = r.get("val_macro_f2") or 0.0
        wf2 = r.get("val_weighted_f2") or 0.0
        fr  = r.get("val_fault_recall") or 0.0
        fa  = r.get("val_false_alarm") or 0.0
        tag = fpga_tag(r["config"])
        print(
            f"  {r['name']:<13} {r['n_params']:>7,} {r['best_epoch']:>7} "
            f"{r['best_val_loss']:>8.4f} {r['val_acc']:>8.4f} "
            f"{mf2:>8.4f} {wf2:>7.4f} "
            f"{fr:>9.4f} {fa:>9.4f} "
            f"{r['runtime_seconds']/60:>7.1f}m {tag:>10}{mark}"
        )
    print(sep)


def print_test_comparison(best_name: str, test_metrics: dict):
    b = OFFICIAL_BASELINE
    t = test_metrics
    delta_acc = t["test_acc"]          - b["test_acc"]
    delta_fr  = t["test_fault_recall"] - b["fault_recall"]
    delta_fa  = t["test_false_alarm"]  - b["false_alarm_rate"]
    delta_mf2 = t["test_macro_f2"]     - b["test_macro_f2"]

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  TEST SET COMPARISON  --  {best_name}  vs  Baseline")
    print(sep)
    print(f"  {'Metric':<22} {'Best Config':>12} {'Baseline':>12} {'Delta':>10}")
    print("  " + "-" * 60)
    print(f"  {'test_acc':<22} {t['test_acc']:>12.4f} {b['test_acc']:>12.4f} {delta_acc:>+10.4f}")
    print(f"  {'test_macro_f2':<22} {t['test_macro_f2']:>12.4f} {b['test_macro_f2']:>12.4f} {delta_mf2:>+10.4f}")
    print(f"  {'fault_recall':<22} {t['test_fault_recall']:>12.4f} {b['fault_recall']:>12.4f} {delta_fr:>+10.4f}")
    print(f"  {'false_alarm_rate':<22} {t['test_false_alarm']:>12.4f} {b['false_alarm_rate']:>12.4f} {delta_fa:>+10.4f}")
    print(sep)


# -- Decision logic -------------------------------------------------------------
def make_decision(best_name: str, best_cfg: dict, test_metrics: dict) -> str:
    b = OFFICIAL_BASELINE
    delta_acc = test_metrics["test_acc"]          - b["test_acc"]
    delta_fr  = test_metrics["test_fault_recall"] - b["fault_recall"]
    delta_mf2 = test_metrics["test_macro_f2"]     - b["test_macro_f2"]

    is_same_arch = (
        best_cfg.get("d_model")        == b["arch"]["d_model"]
        and best_cfg.get("num_layers") == b["arch"]["num_layers"]
        and best_cfg.get("dim_feedforward") == b["arch"]["dim_feedforward"]
        and abs(best_cfg.get("lr", 0) - b["arch"]["lr"]) < 1e-9
    )

    # Regression check: never accept a new model that hurts safety metrics
    fault_recall_ok  = delta_fr  >= -0.001   # allow 0.1 pp regression at most
    false_alarm_ok   = test_metrics["test_false_alarm"] <= b["false_alarm_rate"] + 0.005

    meaningfully_better = (
        (delta_acc > 0.003 or delta_mf2 > 0.003 or delta_fr > 0.001)
        and fault_recall_ok
        and false_alarm_ok
    )

    if is_same_arch:
        return (
            f"KEEP ORIGINAL BASELINE -- best config ({best_name}) is the same "
            f"architecture; confirms baseline is competitive."
        )
    if not fault_recall_ok or not false_alarm_ok:
        return (
            f"KEEP ORIGINAL BASELINE -- {best_name} degrades safety metrics "
            f"(fault_recall D={delta_fr:+.4f}, false_alarm D={test_metrics['test_false_alarm']-b['false_alarm_rate']:+.4f})."
        )
    if not meaningfully_better:
        return (
            f"KEEP ORIGINAL BASELINE -- {best_name} improvement is marginal "
            f"(Dacc={delta_acc:+.4f}, Dmacro_f2={delta_mf2:+.4f}) and does not "
            f"justify replacing the established checkpoint."
        )
    # Improvement is real -- but still flag FPGA cost
    tag = fpga_tag(best_cfg)
    return (
        f"CONSIDER REPLACING with {best_name} "
        f"(Dacc={delta_acc:+.4f}, Dfault_recall={delta_fr:+.4f}, "
        f"Dmacro_f2={delta_mf2:+.4f}). "
        f"FPGA suitability: {tag}. "
        f"Verify Stage 8 resource budget before committing."
    )


# -- Main -----------------------------------------------------------------------
def main():
    tuning_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_dir = Path(WORKING_DIR) / f"tuning_{tuning_ts}"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    log(f"Tuning run directory : {tuning_dir}")
    log(f"Time budget          : {MAX_BUDGET_SECONDS/3600:.1f} h")
    log(f"Configs planned      : {len(TUNING_CONFIGS)}")

    # -- Locate inputs ----------------------------------------------------------
    code_root = locate_source_root(INPUT_ROOT)
    raw_dir   = detect_mafaulda_root(INPUT_ROOT)
    if raw_dir is None:
        raise FileNotFoundError(
            "Cannot find MAFAULDA root -- attach vuxuancu/mafaulda-full."
        )
    log(f"code_root={code_root}")
    log(f"raw_dir  ={raw_dir}")

    os.environ["PROJECT_ROOT"] = WORKING_DIR
    sys.path.insert(0, code_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log(f"device={device}  ({gpu_name})")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # -- Build DataLoaders ONCE -------------------------------------------------
    log("Building DataLoaders (preprocessing runs once; all configs share the tensors)...")
    t0 = time.perf_counter()
    from src.data.make_dataloaders import build_dataloaders

    train_loader, val_loader, test_loader, stats = build_dataloaders(
        data_dir=raw_dir,
        max_files=None,
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        normalization_scope=NORM_SCOPE,
        normalization_method=NORM_METHOD,
    )
    build_secs = time.perf_counter() - t0
    log(f"DataLoaders ready in {build_secs/60:.1f}m")
    log(f"  train={len(train_loader.dataset):,}  "
        f"val={len(val_loader.dataset):,}  "
        f"test={len(test_loader.dataset):,}  windows")

    # Extract train labels for loss function construction
    if hasattr(train_loader.dataset, "tensors"):
        train_labels = train_loader.dataset.tensors[1].numpy()
    else:
        train_labels = np.asarray(train_loader.dataset.targets)

    # One shared criterion (weights are the same for every config since
    # the train split is the same)
    criterion = build_criterion(train_labels, device)
    log(f"  Class weights: {[round(w, 3) for w in criterion.weight.cpu().tolist()]}")

    # -- Config sweep -----------------------------------------------------------
    all_results: list = []
    skipped:     list = []

    for cfg in TUNING_CONFIGS:
        rem = remaining_seconds()
        if rem < MIN_PER_CONFIG_SECONDS:
            log(
                f"[BUDGET] {rem/60:.1f}m remaining < {MIN_PER_CONFIG_SECONDS/60:.0f}m threshold "
                f"-- skipping {cfg['name']} and all subsequent configs."
            )
            skipped.append(cfg["name"])
            continue

        try:
            result = train_one_config(
                cfg, train_loader, val_loader, criterion, device, tuning_dir
            )
            all_results.append(result)
        except Exception as exc:
            log(f"[ERROR] Config {cfg['name']} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            all_results.append({
                "name": cfg["name"], "config": cfg,
                "error": str(exc),
                "n_params": None, "best_epoch": None,
                "best_val_loss": float("inf"), "val_acc": 0.0,
                "val_macro_f2": 0.0, "val_weighted_f2": 0.0,
                "val_fault_recall": 0.0, "val_false_alarm": 0.0,
                "runtime_seconds": 0,
            })

        # Save running summary after every config
        with open(tuning_dir / "tuning_partial_summary.json", "w") as fh:
            json.dump(
                {
                    "completed": [r["name"] for r in all_results if not r.get("error")],
                    "failed":    [r["name"] for r in all_results if r.get("error")],
                    "skipped":   skipped,
                    "results":   all_results,
                    "elapsed_minutes": round(elapsed_seconds() / 60, 1),
                },
                fh, indent=2,
            )

    # -- Select best ------------------------------------------------------------
    valid_results = [r for r in all_results if not r.get("error")]
    if not valid_results:
        log("[FATAL] No configs completed without error.")
        return

    best_result = select_best(valid_results)
    best_name   = best_result["name"]
    best_cfg    = best_result["config"]

    print_summary(valid_results, best_name)
    log(f"\nSelected best config : {best_name}  (by val_macro_f2 then val_acc)")

    # -- Test evaluation -- ONCE for the winner only -----------------------------
    log(f"Running test set evaluation for {best_name}...")
    ckpt = torch.load(best_result["checkpoint_path"], map_location=device)
    set_seed(SEED)
    best_model = TuningTransformer(
        d_model=best_cfg["d_model"],
        nhead=best_cfg["nhead"],
        num_layers=best_cfg["num_layers"],
        dim_feedforward=best_cfg["dim_feedforward"],
        dropout=best_cfg["dropout"],
    ).to(device)
    best_model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, test_cm = eval_loop(best_model, test_loader, criterion, device)
    test_macro_f2_val   = macro_f2(test_cm)
    test_wt_f2_val      = weighted_f2(test_cm)
    test_safety         = safety_metrics(test_cm)
    test_per_cls        = per_class_accuracy(test_cm)

    test_metrics = {
        "test_acc":          float(test_acc),
        "test_macro_f2":     float(test_macro_f2_val),
        "test_weighted_f2":  float(test_wt_f2_val),
        "test_fault_recall": float(test_safety["fault_recall"]),
        "test_false_alarm":  float(test_safety["false_alarm_rate"]),
        "test_safety":       {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                              for k, v in test_safety.items()},
        "test_per_class_acc": test_per_cls,
        "test_confusion":     test_cm.tolist(),
    }

    print_test_comparison(best_name, test_metrics)

    decision = make_decision(best_name, best_cfg, test_metrics)
    log(f"\nDecision: {decision}")

    # -- Final report -----------------------------------------------------------
    peak_mem_gib = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() else None
    )
    final_report = {
        "tuning_run_id":  f"tuning_{tuning_ts}",
        "date":           tuning_ts,
        "total_runtime_minutes": round(elapsed_seconds() / 60, 1),
        "gpu_name":       gpu_name,
        "peak_gpu_memory_gib": peak_mem_gib,
        "fixed_settings": {
            "window_size":    WINDOW_SIZE,
            "stride":         STRIDE,
            "normalization":  f"{NORM_SCOPE}/{NORM_METHOD}",
            "batch_size":     BATCH_SIZE,
            "max_epochs":     MAX_EPOCHS,
            "patience":       PATIENCE,
            "seed":           SEED,
        },
        "official_baseline":    OFFICIAL_BASELINE,
        "configs_attempted":    len(TUNING_CONFIGS),
        "configs_completed":    [r["name"] for r in valid_results],
        "configs_failed":       [r["name"] for r in all_results if r.get("error")],
        "configs_skipped":      skipped,
        "all_val_results":      all_results,
        "best_config":          best_name,
        "best_config_params":   best_result["n_params"],
        "best_val_metrics": {
            "val_acc":       best_result["val_acc"],
            "val_macro_f2":  best_result["val_macro_f2"],
            "best_val_loss": best_result["best_val_loss"],
        },
        "test_metrics_for_best": test_metrics,
        "decision": decision,
    }

    report_path = tuning_dir / "tuning_final_report.json"
    with open(report_path, "w") as fh:
        json.dump(final_report, fh, indent=2)
    log(f"\nFinal report saved -> {report_path}")

    # Also copy to /kaggle/working root for easy download
    import shutil
    shutil.copy(report_path, Path(WORKING_DIR) / "tuning_final_report.json")

    print(f"\n{'=' * 64}")
    print(f"  DECISION")
    print(f"{'=' * 64}")
    print(f"  {decision}")
    print(f"{'=' * 64}")
    print(f"\nTotal runtime: {elapsed_seconds()/60:.1f}m")


if __name__ == "__main__":
    main()
