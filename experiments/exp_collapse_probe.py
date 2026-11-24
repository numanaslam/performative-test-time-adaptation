"""Collapse probe: does PERFORMATIVITY (eps) control collapse once we enter
TTA's fragile regime? Sweeps eps x batch_size at a long horizon with a
collapse-prone optimizer (Adam by default). Tracks both the induced-marginal
entropy H(q) AND BALANCED accuracy (bacc, on class-uniform held-out batches) --
the metric that reveals whether concentration actually harms the classifier or
is a benign, self-consistent equilibrium.

    python -m experiments.exp_collapse_probe --adapter tent --eval_every 50 --out results/probe_tent_bacc.npz
    python -m experiments.exp_collapse_probe --adapter plst --eval_every 50 --out results/probe_plst_bacc.npz
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
    ap.add_argument("--adapter", default="tent", choices=["tent", "plst", "eata_lite", "patta"])
    ap.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eps", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--bs", default="4,8,16,32,64,128")
    ap.add_argument("--eval_every", type=int, default=50, help="balanced-acc probe interval (0=off)")
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    K = num_classes_of(args.which)
    eps_list = parse_list(args.eps, float)
    bs_list = parse_list(args.bs, int)
    pool = CorruptedPool(args.cdir, args.corruption, args.severity, num_classes=K, device=device)
    print(f"device={device} adapter={args.adapter} optim={args.optim} lr={args.lr} steps={args.steps}")
    print(f"eps={eps_list} bs={bs_list} eval_every={args.eval_every}")

    ttc = np.zeros((len(eps_list), len(bs_list)))
    Hq = np.zeros_like(ttc)
    acc = np.zeros_like(ttc)
    bacc = np.zeros_like(ttc)
    rows = []
    for i, e in enumerate(eps_list):
        for j, bs in enumerate(bs_list):
            ad = fresh_adapter(args.adapter, args.ckpt, K, device, lr=args.lr, optim=args.optim)
            log = run_performative_loop(ad, pool, K, e, args.steps, bs, seed=args.seed,
                                        eval_every=args.eval_every, eval_bs=args.eval_bs)
            ttc[i, j], Hq[i, j], acc[i, j], bacc[i, j] = (
                log["ttc"], log["final_Hq"], log["mean_acc"], log["final_bacc"])
            rows.append((e, bs, log["ttc"], log["final_Hq"], log["mean_acc"],
                         log["init_bacc"], log["final_bacc"]))
            print(f"eps={e:.2f} bs={bs:4d}  ttc={log['ttc']:5d}  final_Hq={log['final_Hq']:.3f}  "
                  f"stream_acc={log['mean_acc']:.3f}  bal_acc={log['init_bacc']:.3f}->{log['final_bacc']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, eps=np.array(eps_list), bs=np.array(bs_list),
             ttc=ttc, Hq=Hq, acc=acc, bacc=bacc,
             which=args.which, corruption=args.corruption, severity=args.severity)
    csvp = os.path.splitext(args.out)[0] + ".csv"
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps", "bs", "ttc", "final_Hq", "stream_acc", "init_bacc", "final_bacc"])
        w.writerows(rows)
    print("saved", args.out, csvp)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        im0 = ax[0].imshow(Hq, origin="lower", aspect="auto", cmap="viridis",
                           extent=[0, len(bs_list), 0, len(eps_list)])
        ax[0].set_title("final H(q) (low=concentration)")
        ax[0].set_xlabel("batch-size index (small -> large)")
        ax[0].set_ylabel("eps index (performativity)")
        fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(bacc, origin="lower", aspect="auto", cmap="magma",
                           extent=[0, len(bs_list), 0, len(eps_list)])
        ax[1].set_title("balanced accuracy (low=harm)")
        ax[1].set_xlabel("batch-size index")
        ax[1].set_ylabel("eps index")
        fig.colorbar(im1, ax=ax[1])
        png = os.path.splitext(args.out)[0] + ".png"
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        print("saved", png)
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
