"""Render publication-style phase diagrams from probe_*.npz files.

Fixes the auto-plot (which labels axes by index): real eps/bs ticks, H(q)
normalized by log K so CIFAR-10 and CIFAR-100 are comparable (0 = full
collapse, 1 = uniform), a white collapse-boundary contour, and side-by-side
comparison of several runs (e.g. TENT vs pseudo-label).

  python -m experiments.plot_phase --npz results/probe_tent.npz --label TENT --out results/phase_tent.png
  python -m experiments.plot_phase --npz results/probe_tent.npz results/probe_plst.npz --label TENT PLST --out results/phase_compare.png
  python -m experiments.plot_phase --npz results/probe_plst.npz --metric acc --out results/phase_plst_acc.png
"""
import argparse
import math
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--label", nargs="+", default=None)
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--metric", default="Hq", choices=["Hq", "acc", "bacc", "ttc"])
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="collapse boundary drawn at this fraction of log K (Hq metric)")
    ap.add_argument("--out", default="results/phase.png")
    args = ap.parse_args()

    K = 100 if args.which == "cifar100" else 10
    logK = math.log(K)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = args.label or [os.path.splitext(os.path.basename(p))[0] for p in args.npz]
    n = len(args.npz)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.6), squeeze=False)
    for k, path in enumerate(args.npz):
        d = np.load(path)
        eps, bs = d["eps"], d["bs"]
        M = d[args.metric].astype(float)
        if args.metric == "Hq":
            M = M / logK
            vmin, vmax, cmap, cbar = 0.0, 1.0, "viridis", "H(q) / log K  (low = collapse)"
        elif args.metric == "acc":
            vmin, vmax, cmap, cbar = float(M.min()), float(M.max()), "viridis", "stream accuracy"
        elif args.metric == "bacc":
            vmin, vmax, cmap, cbar = float(M.min()), float(M.max()), "magma", "balanced accuracy (low = harm)"
        else:
            vmin, vmax, cmap, cbar = 0.0, float(M.max()), "magma", "time-to-collapse"

        ax = axes[0][k]
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([str(int(b)) for b in bs])
        ax.set_yticks(range(len(eps)))
        ax.set_yticklabels([f"{e:.1f}" for e in eps])
        ax.set_xlabel("batch size")
        ax.set_ylabel("performativity  eps")
        ax.set_title(labels[k] if k < len(labels) else os.path.basename(path))
        if args.metric == "Hq":
            try:
                ax.contour(M, levels=[args.threshold], colors="white", linewidths=2)
            except Exception:
                pass
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white")
        fig.colorbar(im, ax=ax, label=cbar)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
