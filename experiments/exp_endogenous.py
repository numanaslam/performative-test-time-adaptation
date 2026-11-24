"""Endogenous-feedback experiment: does the model induce its OWN class imbalance and
collapse on a BALANCED, IID stream with NO eps and NO exogenous correlation -- purely
through its confidence-gated replay buffer?

For one endogenous-strength cell `alpha` (fraction of the training batch drawn from the
self-selected buffer), runs each adapter and records balanced accuracy, stream accuracy,
and u_buf (concentration of the buffer's TRUE class marginal = the self-induced imbalance).

Attribution (compare across alpha in the driver):
  alpha=0   train on fresh balanced batches only  -> control (no feedback, no collapse)
  alpha=1   train on the self-selected buffer      -> full endogenous feedback (collapse)
  patta/patta_sev at alpha=1                        -> the derived fix on the clean loop

    python -m experiments.exp_endogenous --which cifar10 --corruption gaussian_noise \
        --alpha 1.0 --bs 16 --steps 2000 --seed 0 --out results/endo/endo_c10_gaussian_noise_a10_s0.npz
"""
import argparse
import math
import os
import numpy as np

from perftta.datasets import CorruptedPool
from perftta.endogenous_loop import run_endogenous_loop
from perftta.utils import get_device
from experiments._common import fresh_adapter, num_classes_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/data/checkpoints/resnet18_cifar10.pt")
    ap.add_argument("--cdir", default="D:/data/cifar10c/CIFAR-10-C")
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--adapters", default="source,tent,eata_lite,patta,patta_sev")
    ap.add_argument("--alpha", type=float, default=1.0)        # endogenous feedback strength (0=control)
    ap.add_argument("--buffer_cap", type=int, default=2000)
    ap.add_argument("--select_frac", type=float, default=0.4)   # confidence threshold tau = frac*lnK
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/endo/endo.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    K = num_classes_of(args.which)
    pool = CorruptedPool(args.cdir, args.corruption, args.severity, num_classes=K, device=device)
    tau = args.select_frac * math.log(K)
    names = [a.strip() for a in args.adapters.split(",") if a.strip()]
    print(f"device={device} which={args.which} corr={args.corruption} alpha={args.alpha} "
          f"buf={args.buffer_cap} bs={args.bs} tau={tau:.3f} seed={args.seed}")

    bacc, sacc, ubuf, hbuf = [], [], [], []
    for name in names:
        ad = fresh_adapter(name, args.ckpt, K, device, lr=args.lr, optim=args.optim)
        log = run_endogenous_loop(ad, pool, K, args.steps, args.bs, buffer_cap=args.buffer_cap,
                                  alpha=args.alpha, select_tau=tau, seed=args.seed,
                                  eval_every=args.eval_every, eval_bs=args.eval_bs)
        bacc.append(log["final_bacc"]); sacc.append(log["final_acc"])
        ubuf.append(log["u_buf"]); hbuf.append(log["final_Hbuf"])
        print(f"  [{name:10s}] bal={log['init_bacc']:.3f}->{log['final_bacc']:.3f}  "
              f"stream={log['final_acc']:.3f}  u_buf={log['u_buf']:.3f}  buf={log['final_buf_size']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, which=args.which, corruption=args.corruption, severity=args.severity,
             alpha=args.alpha, buffer_cap=args.buffer_cap, bs=args.bs, seed=args.seed,
             adapters=np.array(names), final_bacc=np.array(bacc), final_acc=np.array(sacc),
             u_buf=np.array(ubuf), final_Hbuf=np.array(hbuf))
    print("saved", args.out)


if __name__ == "__main__":
    main()
