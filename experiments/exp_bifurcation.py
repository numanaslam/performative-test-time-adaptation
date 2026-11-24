"""E1 - Collapse bifurcation map (the headline figure).

Sweeps performativity strength eps x learning rate lr; per cell runs the
performative loop and records the final induced-marginal entropy H(q), the
time-to-collapse, and mean online accuracy. A sharp eps-dependent boundary
between stable (high H(q)) and collapsed (low H(q)) regions is the central
prediction (Theorem T2).

    python -m experiments.exp_bifurcation --ckpt D:/data/checkpoints/resnet18_cifar10.pt \
        --cdir D:/data/cifar10c/CIFAR-10-C --corruption gaussian_noise --severity 5
"""
import argparse
import csv
import os
import numpy as np

from perftta.datasets import CorruptedPool
from perftta.loop import run_performative_loop
from perftta.utils import get_device
from experiments._common import fresh_adapter, num_classes_of


def parse_list(s, typ):
    return [typ(v) for v in s.split(",") if v != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/data/checkpoints/resnet18_cifar10.pt")
    ap.add_argument("--cdir", default="D:/data/cifar10c/CIFAR-10-C")
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--adapter", default="tent")
    ap.add_argument("--eps", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--lr", default="0.0001,0.00025,0.0005,0.001,0.0025,0.005")
    ap.add_argument("--optim", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bifurcation.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    K = num_classes_of(args.which)
    eps_list = parse_list(args.eps, float)
    lr_list = parse_list(args.lr, float)
    print(f"device={device} classes={K} eps={eps_list} lr={lr_list}")

    pool = CorruptedPool(args.cdir, args.corruption, args.severity, num_classes=K, device=device)

    Hq = np.zeros((len(eps_list), len(lr_list)))
    ttc = np.zeros_like(Hq)
    acc = np.zeros_like(Hq)
    rows = []
    for i, e in enumerate(eps_list):
        for j, lr in enumerate(lr_list):
            ad = fresh_adapter(args.adapter, args.ckpt, K, device, lr=lr, optim=args.optim)
            log = run_performative_loop(ad, pool, K, e, args.steps, args.bs, seed=args.seed)
            Hq[i, j], ttc[i, j], acc[i, j] = log["final_Hq"], log["ttc"], log["mean_acc"]
            rows.append((e, lr, log["final_Hq"], log["ttc"], log["mean_acc"]))
            print(f"eps={e:.2f} lr={lr:.4g}  final_Hq={log['final_Hq']:.3f}  "
                  f"ttc={log['ttc']}  acc={log['mean_acc']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, eps=np.array(eps_list), lr=np.array(lr_list), Hq=Hq, ttc=ttc, acc=acc)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps", "lr", "final_Hq", "ttc", "mean_acc"])
        w.writerows(rows)
    print("saved", args.out, csv_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        im0 = ax[0].imshow(Hq, origin="lower", aspect="auto", cmap="viridis",
                           extent=[0, len(lr_list), 0, len(eps_list)])
        ax[0].set_title("final H(q)  (low = collapse)")
        ax[0].set_xlabel("lr index"); ax[0].set_ylabel("eps index")
        fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(ttc, origin="lower", aspect="auto", cmap="magma",
                           extent=[0, len(lr_list), 0, len(eps_list)])
        ax[1].set_title("time-to-collapse  (high = stable)")
        ax[1].set_xlabel("lr index"); ax[1].set_ylabel("eps index")
        fig.colorbar(im1, ax=ax[1])
        png = os.path.splitext(args.out)[0] + ".png"
        fig.tight_layout(); fig.savefig(png, dpi=120)
        print("saved", png)
    except Exception as ex:  # plotting is optional
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
