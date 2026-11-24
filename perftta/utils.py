import os
import time
import random
import numpy as np
import torch

# CIFAR normalization (shared by CIFAR-10/100 and their -C variants for this scaffold)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _cuda_courtesy(device):
    """Be a good neighbor on a shared GPU. Controlled entirely by env vars, so
    default (single-user) behavior is unchanged unless they are set:
      PERFTTA_GPU_MIN_FREE_MB    wait until >= this many MB are free before starting
      PERFTTA_GPU_WAIT_TIMEOUT_S max seconds to wait, then proceed anyway (default 7200)
      PERFTTA_GPU_POLL_S         poll interval while waiting (default 30)
      PERFTTA_GPU_MEM_FRACTION   cap THIS process to this fraction of total VRAM
    Never raises: courtesy must not break a run."""
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        min_free = float(os.environ.get("PERFTTA_GPU_MIN_FREE_MB", "0"))
        if min_free > 0:
            timeout = float(os.environ.get("PERFTTA_GPU_WAIT_TIMEOUT_S", "7200"))
            poll = float(os.environ.get("PERFTTA_GPU_POLL_S", "30"))
            waited = 0.0
            while True:
                free_mb = torch.cuda.mem_get_info(idx)[0] / 1e6
                if free_mb >= min_free:
                    break
                if waited >= timeout:
                    print(f"[courtesy] {free_mb:.0f} MB free < {min_free:.0f} after "
                          f"{waited:.0f}s; proceeding with capped footprint.")
                    break
                print(f"[courtesy] GPU busy ({free_mb:.0f} MB free, need {min_free:.0f}); "
                      f"waiting {poll:.0f}s so other jobs are undisturbed...")
                time.sleep(poll)
                waited += poll
        frac = float(os.environ.get("PERFTTA_GPU_MEM_FRACTION", "0"))
        if frac > 0:
            torch.cuda.set_per_process_memory_fraction(min(max(frac, 0.05), 1.0), idx)
            print(f"[courtesy] capped this process to {frac:.2f} of GPU memory.")
    except Exception as e:
        print("[courtesy] skipped:", e)


def get_device(pref=None):
    """cuda -> mps -> cpu, unless a preference string is given. Applies GPU
    courtesy (env-controlled) whenever a CUDA device is selected."""
    if pref:
        dev = torch.device(pref)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    if dev.type == "cuda":
        _cuda_courtesy(dev)
    return dev


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax_entropy(logits):
    """Per-sample Shannon entropy of the softmax (shape [N])."""
    p = logits.softmax(1)
    logp = logits.log_softmax(1)
    return -(p * logp).sum(1)


def marginal_entropy(prob_mean, eps=1e-12):
    """Entropy of a (differentiable) marginal distribution vector."""
    p = prob_mean.clamp_min(eps)
    return -(p * p.log()).sum()
