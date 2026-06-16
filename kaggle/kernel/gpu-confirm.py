"""
GPU-ONLY confirmation for the FPGA fault-diagnosis Transformer.

Goal
----
After manually selecting "GPU T4 x2" in the Kaggle UI, confirm that:
  * an accelerator is assigned and it is a Tesla T4,
  * PyTorch can actually execute CUDA kernels on it,
  * the real project model runs a forward pass on the GPU,
  * a couple of real training steps (backward + optimizer step) run on the GPU.

Strict rules
------------
  * NO CPU fallback. If CUDA is unavailable, not a T4, or cannot run a kernel,
    the script aborts with a non-zero exit (kernel shows ERROR).
  * This is NOT training. It runs the real TransformerBaseline architecture on
    synthetic tensors of the correct shape (batch, WINDOW_SIZE, INPUT_CHANNELS),
    so it is a few seconds of pure GPU verification with no data loading.

Attached input: dataset  mehrdadsheikhabbasi/fpga-fault-src  (the project `src/`).
Output: /kaggle/working/gpu_confirm_report.json
"""

import os
import sys
import glob
import json
import traceback

WORKING_DIR = "/kaggle/working"
INPUT_ROOT = "/kaggle/input"

report = {
    "test": "gpu_only_confirmation",
    "passed": False,
    "stage": "init",
    "errors": [],
}


def log(msg):
    print(msg, flush=True)


def _save():
    try:
        os.makedirs(WORKING_DIR, exist_ok=True)
        with open(os.path.join(WORKING_DIR, "gpu_confirm_report.json"), "w") as f:
            json.dump(report, f, indent=2)
    except Exception as exc:  # pragma: no cover
        log(f"[GPU][WARN] could not write report: {exc}")


def fail(stage, exc):
    report["stage"] = stage
    report["errors"].append(f"{stage}: {type(exc).__name__}: {exc}")
    log(f"\n[GPU][FAIL] stage={stage}: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    _save()
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1) Locate the project source and set a writable PROJECT_ROOT (before import)
# ---------------------------------------------------------------------------
try:
    report["stage"] = "locate_source"
    hits = glob.glob(os.path.join(INPUT_ROOT, "**", "src", "config.py"), recursive=True)
    log(f"[GPU] src/config.py candidates = {hits}")
    if not hits:
        raise FileNotFoundError("Could not find src/config.py under /kaggle/input.")
    code_root = os.path.dirname(os.path.dirname(hits[0]))
    sys.path.insert(0, code_root)
    os.environ["PROJECT_ROOT"] = WORKING_DIR
    report["code_root"] = code_root
except Exception as exc:
    fail("locate_source", exc)


# ---------------------------------------------------------------------------
# 2) GPU checks: present? is it a T4? can it run a CUDA kernel? (no fallback)
# ---------------------------------------------------------------------------
try:
    report["stage"] = "gpu_checks"
    import torch
    from src import config as cfg
    from src.models.transformer_baseline import TransformerBaseline

    report["torch_version"] = torch.__version__

    cuda_available = torch.cuda.is_available()
    report["cuda_available"] = bool(cuda_available)
    if not cuda_available:
        raise RuntimeError(
            "torch.cuda.is_available() is False. No GPU assigned. "
            "GPU-only test aborts (no CPU fallback)."
        )

    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    report["gpu_name"] = gpu_name
    report["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
    report["torch_supported_arch_list"] = torch.cuda.get_arch_list()
    is_t4 = "T4" in gpu_name
    report["gpu_is_t4"] = bool(is_t4)
    log(f"[GPU] name={gpu_name} capability=sm_{cap[0]}{cap[1]} is_t4={is_t4}")
    log(f"[GPU] torch={torch.__version__} supported_arch={torch.cuda.get_arch_list()}")

    if not is_t4:
        raise RuntimeError(
            f"Assigned GPU is '{gpu_name}', NOT a Tesla T4. Aborting (no CPU "
            "fallback). Re-select 'GPU T4 x2' in the Kaggle UI and re-run."
        )

    # Actually execute a CUDA kernel to prove the build supports this device.
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    report["cuda_kernel_exec_ok"] = True
    report["matmul_checksum"] = float(c.sum().item())
    log("[GPU] CUDA kernel execution OK (1024x1024 matmul on cuda).")
except Exception as exc:
    fail("gpu_checks", exc)


# ---------------------------------------------------------------------------
# 3) Real model forward pass on the GPU (synthetic input of the correct shape)
# ---------------------------------------------------------------------------
try:
    report["stage"] = "forward_pass"
    torch.manual_seed(cfg.RANDOM_SEED)
    device = torch.device("cuda")

    model = TransformerBaseline().to(device)
    report["num_model_params"] = int(sum(p.numel() for p in model.parameters()))

    batch_size = 8
    x = torch.randn(batch_size, cfg.INPUT_LENGTH, cfg.INPUT_CHANNELS, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(x)

    report["forward_input_shape"] = list(x.shape)
    report["forward_output_shape"] = list(logits.shape)
    report["model_on_cuda"] = bool(next(model.parameters()).is_cuda)
    report["output_on_cuda"] = bool(logits.is_cuda)
    log(f"[GPU] forward {list(x.shape)} -> {list(logits.shape)} "
        f"(output_on_cuda={logits.is_cuda})")
    if logits.shape[1] != cfg.NUM_CLASSES:
        raise ValueError(
            f"Output dim {logits.shape[1]} != NUM_CLASSES {cfg.NUM_CLASSES}"
        )
except Exception as exc:
    fail("forward_pass", exc)


# ---------------------------------------------------------------------------
# 4) Two real training steps on the GPU (forward + backward + optimizer step)
# ---------------------------------------------------------------------------
try:
    report["stage"] = "training_batches"
    import torch.nn as nn

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    n_batches = 2
    losses = []
    for i in range(n_batches):
        x = torch.randn(8, cfg.INPUT_LENGTH, cfg.INPUT_CHANNELS, device=device)
        y = torch.randint(0, cfg.NUM_CLASSES, (8,), device=device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.item()))
        log(f"[GPU] train batch {i + 1}/{n_batches} loss={loss.item():.4f} "
            f"logits_on_cuda={logits.is_cuda}")

    report["train_batches_run"] = n_batches
    report["train_batch_losses"] = losses
    report["gpu_max_mem_allocated_mb"] = round(
        torch.cuda.max_memory_allocated() / 1e6, 2
    )
except Exception as exc:
    fail("training_batches", exc)


# ---------------------------------------------------------------------------
report["passed"] = True
report["stage"] = "complete"
_save()
log("\n[GPU][PASS] GPU-only confirmation passed (forward + 2 training steps on T4).")
log(json.dumps(report, indent=2))
