"""Cross-corruption / cross-dataset generalization table from collapse-cell
rescue runs.

Each rescue npz holds the per-run scalar verdict (TENT vs PATTA balanced
accuracy at the collapse cell), tagged with its dataset + corruption. This
groups them by (dataset, corruption), averages over seeds, and reports whether
PATTA preserves balanced accuracy where TENT collapses -- across every setting.

    python -m experiments.summarize_generalization --glob "results/gen/rescue_*.npz" --out results/generalization.csv
"""
import argparse
import csv
import glob
import os
import re
import numpy as np


def tag_of(path, d):
    """(which, corruption) from npz keys, falling back to the filename."""
    which = str(d["which"]) if "which" in d.files else None
    corr = str(d["corruption"]) if "corruption" in d.files else None
    if which and corr:
        return which, corr
    m = re.match(r"rescue_(c10|c100|cifar10|cifar100)_(.+)_s\d+\.npz",
                 os.path.basename(path))
    if m:
        return m.group(1), m.group(2)
    return which or "?", corr or os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/gen/rescue_*.npz")
    ap.add_argument("--out", default="results/generalization.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"[error] no files match: {args.glob}")
        raise SystemExit(2)

    groups = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        key = tag_of(p, d)
        groups.setdefault(key, []).append(
            (float(d["tent_final_bacc"]), float(d["patta_final_bacc"]),
             float(d["tent_final_acc"]), float(d["patta_final_acc"])))

    rows = []
    print(f"{'dataset':8s} {'corruption':18s} {'n':>2s}  "
          f"{'TENT bal':>12s} {'PATTA bal':>12s} {'margin':>8s} {'wins':>5s} "
          f"{'TENT str':>9s} {'PATTA str':>9s}")
    all_margin, all_win, all_n = [], 0, 0
    for (which, corr) in sorted(groups):
        arr = np.array(groups[(which, corr)])
        tb, pb, ts, ps = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        margin = pb - tb
        wins = int(np.sum(pb >= tb))
        n = len(arr)
        all_margin.extend(margin.tolist()); all_win += wins; all_n += n
        print(f"{which:8s} {corr:18s} {n:2d}  "
              f"{tb.mean():.3f}+/-{tb.std():.3f} {pb.mean():.3f}+/-{pb.std():.3f} "
              f"{margin.mean():+.3f} {wins:2d}/{n:<2d} "
              f"{ts.mean():9.3f} {ps.mean():9.3f}")
        rows.append([which, corr, n, f"{tb.mean():.4f}", f"{tb.std():.4f}",
                     f"{pb.mean():.4f}", f"{pb.std():.4f}", f"{margin.mean():.4f}",
                     f"{wins}/{n}", f"{ts.mean():.4f}", f"{ps.mean():.4f}"])

    am = np.array(all_margin)
    print(f"\nOVERALL: {all_win}/{all_n} runs PATTA >= TENT on balanced acc;  "
          f"mean margin {am.mean():+.3f} +/- {am.std():.3f}")
    print("VERDICT:", "GENERALIZES" if all_win == all_n and am.mean() > 0.05
          else ("MOSTLY GENERALIZES" if all_win > 0.8 * all_n else "DOES NOT GENERALIZE"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "corruption", "n_seeds", "tent_bal_mean", "tent_bal_std",
                    "patta_bal_mean", "patta_bal_std", "margin_mean", "wins",
                    "tent_stream_mean", "patta_stream_mean"])
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
