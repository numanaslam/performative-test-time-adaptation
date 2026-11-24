"""Quantitative bifurcation test: measure the ORDER PARAMETER as a function of
performativity eps, from the HEALTHY (eps=0-adapted) state, on the real deep network.

Theory (Appendix B) predicts a pitchfork: the induced-marginal concentration
  s(theta) = 1 - H(q)/log K
should stay ~0 (balanced) for eps < eps*, then rise sharply above eps* -- and eps*
should INCREASE with batch size B. This script measures s(eps) directly (the bifurcation
diagram) and the balanced accuracy at each eps, so the stability transition and the
accuracy collapse can be checked to coincide at the same eps*(B).

Protocol per batch size B: adapt eps=0 for `warmup` steps to reach the healthy balanced
state; snapshot the BN-affine params; then for each eps, restore the snapshot (fresh
optimizer) and run `steps` closed-loop steps, recording the tail concentration s_final
and the balanced accuracy.

    python -m experiments.exp_critical --which cifar10 --corruption gaussian_noise \
        --bs 16 --warmup 200 --steps 150 --seed 0 --out results/crit/crit_c10_gaussian_noise_bs16_s0.npz
"""
import argparse
import math
import os
import numpy as np
import torch

from perftta.datasets import CorruptedPool
from perftta.tta import bn_params
from perftta.utils import get_device
from perftta.metrics import pred_hist, entropy_of_dist, accuracy
from experiments._common import fresh_adapter, num_classes_of


def run_eps(ad, pool, K, eps, steps, bs, uniform, Hmax, seed, eval_bs):
    g = torch.Generator().manual_seed(seed)
    eg = torch.Generator().manual_seed(seed + 97)
    prior = uniform.clone()
    svals = []
    for t in range(steps):
        x, _ = pool.sample(prior, bs, generator=g)
        with torch.no_grad():
            logits = ad.predict(x)
        hist = pred_hist(logits.detach().cpu(), K)            # induced marginal q(theta_t)
        svals.append(1.0 - entropy_of_dist(hist) / Hmax)       # order parameter (concentration)
        ad.update(x)
        prior = (1.0 - eps) * uniform + eps * hist             # performative induction
    tail = svals[-max(5, steps // 4):]
    s_final = float(sum(tail) / len(tail))
    xb, yb = pool.sample(uniform, eval_bs, generator=eg)
    with torch.no_grad():
        bacc = accuracy(ad.predict(xb), yb)
    return s_final, bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/data/checkpoints/resnet18_cifar10.pt")
    ap.add_argument("--cdir", default="D:/data/cifar10c/CIFAR-10-C")
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--eps_grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/crit/crit.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    K = num_classes_of(args.which)
    Hmax = math.log(K)
    uniform = torch.ones(K) / K
    pool = CorruptedPool(args.cdir, args.corruption, args.severity, num_classes=K, device=device)
    eps_list = [float(e) for e in args.eps_grid.split(",") if e.strip() != ""]

    ad = fresh_adapter("tent", args.ckpt, K, device, lr=args.lr, optim=args.optim)
    # warmup at eps=0 (balanced batches) -> healthy balanced state
    gw = torch.Generator().manual_seed(args.seed + 1)
    for _ in range(args.warmup):
        x, _ = pool.sample(uniform, args.bs, generator=gw)
        ad.update(x)
    warmed = [p.detach().clone() for p in bn_params(ad.model)]
    params = bn_params(ad.model)
    print(f"device={device} which={args.which} corr={args.corruption} bs={args.bs} "
          f"warmup={args.warmup} seed={args.seed}  (K={K}, Hmax={Hmax:.3f})")

    s_finals, baccs = [], []
    for eps in eps_list:
        for p, w in zip(params, warmed):               # restore the healthy state
            p.data.copy_(w)
        ad.opt = torch.optim.Adam(params, lr=args.lr) if args.optim == "adam" \
            else torch.optim.SGD(params, lr=args.lr, momentum=0.0)   # fresh optimizer moments
        sf, ba = run_eps(ad, pool, K, eps, args.steps, args.bs, uniform, Hmax,
                         args.seed + int(eps * 1000) + 3, args.eval_bs)
        s_finals.append(sf); baccs.append(ba)
        print(f"  eps={eps:.2f}  s_final(concentration)={sf:.3f}  balanced_acc={ba:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, which=args.which, corruption=args.corruption, severity=args.severity,
             bs=args.bs, seed=args.seed, eps=np.array(eps_list),
             s_final=np.array(s_finals), bacc=np.array(baccs), Hmax=Hmax)
    print("saved", args.out)


if __name__ == "__main__":
    main()
