"""E2 - PATTA rescue / kill experiment.

At an (eps, lr) above the naive collapse boundary, does the performativity-aware
update (PATTA) keep the induced marginal spread out and accuracy up where TENT
collapses? This is the method's go/no-go test (Theorem T3).

    python -m experiments.exp_patta_rescue --ckpt D:/data/checkpoints/resnet18_cifar10.pt \
        --cdir D:/data/cifar10c/CIFAR-10-C --eps 0.8 --lr 0.0025
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
    ap.add_argument("--eps", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=0.0025)
    ap.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/patta_rescue.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    K = num_classes_of(args.which)
    pool = CorruptedPool(args.cdir, args.corruption, args.severity, num_classes=K, device=device)

    runs = {}
    for name, kw in [("tent", {}), ("patta", {"lam0": args.lam0})]:
        ad = fresh_adapter(name, args.ckpt, K, device, lr=args.lr, optim=args.optim, **kw)
        log = run_performative_loop(ad, pool, K, args.eps, args.steps, args.bs, seed=args.seed,
                                    eval_every=args.eval_every, eval_bs=args.eval_bs)
        runs[name] = log
        print(f"[{name:5s}] final_Hq={log['final_Hq']:.3f}  ttc={log['ttc']}  "
              f"stream_acc={log['final_acc']:.3f}  bal_acc={log['init_bacc']:.3f}->{log['final_bacc']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out, eps=args.eps, lr=args.lr, bs=args.bs, seed=args.seed,
        which=args.which, corruption=args.corruption, severity=args.severity,
        tent_Hq=np.array(runs["tent"]["Hq"]), patta_Hq=np.array(runs["patta"]["Hq"]),
        tent_acc=np.array(runs["tent"]["acc"]), patta_acc=np.array(runs["patta"]["acc"]),
        tent_share=np.array(runs["tent"]["share"]), patta_share=np.array(runs["patta"]["share"]),
        # scalars for multi-seed aggregation (the balanced-accuracy verdict)
        tent_final_bacc=runs["tent"]["final_bacc"], patta_final_bacc=runs["patta"]["final_bacc"],
        tent_init_bacc=runs["tent"]["init_bacc"], patta_init_bacc=runs["patta"]["init_bacc"],
        tent_final_acc=runs["tent"]["final_acc"], patta_final_acc=runs["patta"]["final_acc"],
        tent_final_Hq=runs["tent"]["final_Hq"], patta_final_Hq=runs["patta"]["final_Hq"],
    )
    print("saved", args.out)

    rescued = runs["patta"]["final_Hq"] > runs["tent"]["final_Hq"] + 0.2
    bal_win = runs["patta"]["final_bacc"] >= runs["tent"]["final_bacc"]
    print("KILL-CRITERION (rescue):", "PASS" if (rescued and bal_win) else "INCONCLUSIVE")
    print(f"  rescued diversity: {rescued} | balanced-acc preserved/raised: {bal_win} "
          f"(tent bal {runs['tent']['final_bacc']:.3f} vs patta bal {runs['patta']['final_bacc']:.3f})")


if __name__ == "__main__":
    main()
