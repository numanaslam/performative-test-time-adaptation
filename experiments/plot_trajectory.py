"""Quick local PNG preview of the trajectory CSVs written by summarize_trajectory.
Optional sanity check on the GPU box; the paper figure is rebuilt in pgfplots from
the same CSVs. Needs matplotlib.

    python -m experiments.plot_trajectory --csvdir results/trajectory_csv --which cifar10 \
        --corruption gaussian_noise --severity 5 --out results/trajectory_csv/traj_c10_gauss.png
"""
import argparse
import csv
import glob
import os


def read_csv(path):
    with open(path) as fh:
        r = csv.DictReader(fh)
        rows = list(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvdir", default="results/trajectory_csv")
    ap.add_argument("--which", default="cifar10")
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--lambda_hist", action="store_true",
                    help="plot the lambda_t histogram instead of trajectories")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.lambda_hist:
        rows = read_csv(os.path.join(args.csvdir, "lambda_ratio_hist.csv"))
        left = [float(r["bin_left"]) for r in rows]
        width = float(rows[0]["bin_right"]) - float(rows[0]["bin_left"])
        count = [int(r["count"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(left, count, width=width, align="edge", color="#4C78A8", edgecolor="white")
        ax.axvline(0.774, color="#E45756", ls="--", lw=1.5, label="bound (p=0.5) 0.774")
        ax.axvline(0.599, color="#F58518", ls=":", lw=1.5, label="bound (p=1) 0.599")
        ax.set_xlabel(r"$\lambda_t/\lambda_0$ on a balanced population ($K\gg B$)")
        ax.set_ylabel("steps"); ax.set_xlim(0, 1); ax.legend(fontsize=8)
        out = args.out or os.path.join(args.csvdir, "lambda_hist.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); print("saved", out)
        return

    pat = os.path.join(args.csvdir,
                       f"traj_{args.which}_{args.corruption}_sev{args.severity}_*.csv")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"no trajectory CSVs match {pat}")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for f in files:
        tag = os.path.basename(f).replace(
            f"traj_{args.which}_{args.corruption}_sev{args.severity}_", "").replace(".csv", "")
        rows = read_csv(f)
        step = [int(r["step"]) for r in rows]
        mean = [float(r["bacc_mean"]) for r in rows]
        ax.plot(step, mean, lw=1.8, label=tag)
    ax.set_xlabel("adaptation step"); ax.set_ylabel("balanced accuracy")
    ax.set_title(f"{args.which} / {args.corruption} / sev{args.severity}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out = args.out or os.path.join(args.csvdir,
                                   f"traj_{args.which}_{args.corruption}_sev{args.severity}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); print("saved", out)


if __name__ == "__main__":
    main()
