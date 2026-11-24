"""Baseline comparison at the collapse cell: does PATTA's DERIVED, adaptive
marginal-entropy correction beat the existing anti-collapse family?

Runs the performative loop for each adapter at one (eps, batch) cell and records
final balanced accuracy, stream accuracy, and H(q). Adapters:
  source      - frozen (no adaptation floor)
  tent        - entropy min (no collapse defense)
  eata_lite   - reliable-sample selection
  sar         - selection + sharpness-aware minimization
  balance     - FIXED-weight marginal-entropy (heuristic class balance)
  patta       - ours: adaptive, stability-derived marginal-entropy

    python -m experiments.exp_baselines --which cifar10 --corruption gaussian_noise \
        --eps 1.0 --bs 16 --steps 1500 --seed 0 --out results/base/base_c10_gaussian_noise_s0.npz
"""
import argparse
import os
import numpy as np

from perftta.datasets import CorruptedPool
from perftta.loop import run_performative_loop
from perftta.utils import get_device
from experiments._common import fresh_adapter, num_classes_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/data/checkpoints/resnet18_cifar10.pt")
    ap.add_argument("--cdir", default="D:/data/cifar10c/CIFAR-10-C")
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--adapters", default="source,tent,eata_lite,sar,balance,patta")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/base/baselines.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    K = num_classes_of(args.which)
    pool = CorruptedPool(args.cdir, args.corruption, args.severity, num_classes=K, device=device)
    names = [a.strip() for a in args.adapters.split(",") if a.strip()]
    print(f"device={device} which={args.which} corr={args.corruption} eps={args.eps} bs={args.bs} seed={args.seed}")

    bacc, sacc, hq, ibacc = [], [], [], []
    for name in names:
        ad = fresh_adapter(name, args.ckpt, K, device, lr=args.lr, optim=args.optim)
        log = run_performative_loop(ad, pool, K, args.eps, args.steps, args.bs, seed=args.seed,
                                    eval_every=args.eval_every, eval_bs=args.eval_bs)
        bacc.append(log["final_bacc"]); sacc.append(log["final_acc"])
        hq.append(log["final_Hq"]); ibacc.append(log["init_bacc"])
        print(f"  [{name:10s}] bal={log['init_bacc']:.3f}->{log['final_bacc']:.3f}  "
              f"stream={log['final_acc']:.3f}  H(q)={log['final_Hq']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, which=args.which, corruption=args.corruption, severity=args.severity,
             seed=args.seed, eps=args.eps, bs=args.bs,
             adapters=np.array(names), final_bacc=np.array(bacc),
             final_acc=np.array(sacc), final_Hq=np.array(hq), init_bacc=np.array(ibacc))
    print("saved", args.out)


if __name__ == "__main__":
    main()
