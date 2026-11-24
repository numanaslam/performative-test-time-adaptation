"""Logic smoke test - NO dataset or checkpoint needed. Validates that the
performative loop runs for every adapter and produces finite metrics. Fast,
CPU-only; run it first on any new machine.

    python -m experiments.smoke_test
"""
import torch

from perftta.models import resnet18
from perftta.tta import build_adapter
from perftta.loop import run_performative_loop


class FakePool:
    """Random tensors standing in for a CIFAR-C pool (exercises code paths only)."""

    def __init__(self, K=10, n=512):
        self.K = K
        self.X = torch.randn(n, 3, 32, 32)
        self.y = torch.randint(0, K, (n,))
        self.by_class = [(self.y == c).nonzero(as_tuple=True)[0] for c in range(K)]

    def sample(self, prior, bs, generator=None):
        idx = torch.randint(0, len(self.y), (bs,), generator=generator)
        return self.X[idx], self.y[idx]


def main():
    K = 10
    pool = FakePool(K)
    for name in ["source", "tent", "eata_lite", "patta"]:
        model = resnet18(K)
        ad = build_adapter(name, model, num_classes=K, lr=1e-3, optim="sgd")
        log = run_performative_loop(ad, pool, K, eps=0.8, steps=12, batch_size=32, seed=0)
        assert len(log["acc"]) == 12
        for key in ["final_Hq", "ttc", "mean_acc", "final_acc"]:
            v = log[key]
            assert v == v, f"{name}:{key} is NaN"
        print(f"[ok] {name:10s} mean_acc={log['mean_acc']:.3f} "
              f"final_Hq={log['final_Hq']:.3f} ttc={log['ttc']}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
