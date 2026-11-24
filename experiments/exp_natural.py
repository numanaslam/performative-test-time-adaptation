"""NATURAL-feedback experiment: does the performative collapse arise WITHOUT the injected
eps knob -- from a realistic correlated stream plus the adapter's own confidence selection?

For one (stream correlation rho) cell, runs each adapter on an exogenous correlated stream
(streams.py) and records balanced accuracy, stream accuracy, and the natural performativity
pressure u_sel = (logK - H(q_sel))/logK measured on the confidence-selected marginal.

Attribution (compare across rho and adapters in the driver):
  rho=0, tent          iid, adapt-on-all        -> control (no collapse)
  rho high, tent       correlated, adapt-on-all -> correlation alone
  rho high, eata/sar   correlated + selection   -> the NATURAL performative loop (collapse)
  rho high, patta      correlated + correction  -> the derived fix still works

    python -m experiments.exp_natural --which cifar10 --corruption gaussian_noise \
        --rho 0.99 --bs 16 --steps 1500 --seed 0 --out results/natural/nat_c10_gaussian_noise_rho099_s0.npz
"""
import argparse
import math
import os
import numpy as np

from perftta.datasets import CorruptedPool
from perftta.streams import CorrelatedStream
from perftta.natural_loop import run_natural_loop
from perftta.utils import get_device
from experiments._common import fresh_adapter, num_classes_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/data/checkpoints/resnet18_cifar10.pt")
    ap.add_argument("--cdir", default="D:/data/cifar10c/CIFAR-10-C")
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--adapters", default="source,tent,eata_lite,sar,patta,patta_sev")
    ap.add_argument("--rho", type=float, default=0.99)        # stream correlation (0 = iid)
    ap.add_argument("--select_frac", type=float, default=0.4)  # confidence threshold tau = frac*lnK
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/natural/natural.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    K = num_classes_of(args.which)
    pool = CorruptedPool(args.cdir, args.corruption, args.severity, num_classes=K, device=device)
    tau = args.select_frac * math.log(K)
    names = [a.strip() for a in args.adapters.split(",") if a.strip()]
    print(f"device={device} which={args.which} corr={args.corruption} rho={args.rho} "
          f"bs={args.bs} tau={tau:.3f} seed={args.seed}")

    bacc, sacc, usel, hqsel, selfrac = [], [], [], [], []
    for name in names:
        ad = fresh_adapter(name, args.ckpt, K, device, lr=args.lr, optim=args.optim)
        stream = CorrelatedStream(pool, K, rho=args.rho, seed=args.seed)   # same seed -> same stream
        log = run_natural_loop(ad, stream, K, args.steps, args.bs, select_tau=tau,
                               seed=args.seed, eval_every=args.eval_every, eval_bs=args.eval_bs)
        bacc.append(log["final_bacc"]); sacc.append(log["final_acc"])
        usel.append(log["u_sel"]); hqsel.append(log["final_Hqsel"]); selfrac.append(log["final_sel_frac"])
        print(f"  [{name:10s}] bal={log['init_bacc']:.3f}->{log['final_bacc']:.3f}  "
              f"stream={log['final_acc']:.3f}  u_sel={log['u_sel']:.3f}  sel_frac={log['final_sel_frac']:.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, which=args.which, corruption=args.corruption, severity=args.severity,
             rho=args.rho, bs=args.bs, seed=args.seed, select_frac=args.select_frac,
             adapters=np.array(names), final_bacc=np.array(bacc), final_acc=np.array(sacc),
             u_sel=np.array(usel), final_Hqsel=np.array(hqsel), final_sel_frac=np.array(selfrac))
    print("saved", args.out)


if __name__ == "__main__":
    main()
