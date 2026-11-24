"""Aggregate exp_trajectory runs across seeds into plot-ready CSVs.

For each (which, corruption, severity) it writes, per configuration, a trajectory CSV
(step, balanced-accuracy mean and std over seeds) and reports the onset step -- the
first eval step where balanced accuracy drops below 90% of its initial probe. For the
balanced-population lambda_t configuration it summarizes the gate distribution and the
fraction of steps at or above the finite-batch lower bound (1 - log B / log K)**p, the
check behind Section 6.8.1.

    python -m experiments.summarize_trajectory --glob "results/trajectory/*.npz" --outdir results/trajectory_csv
"""
import argparse
import csv
import glob
import math
import os
from collections import defaultdict
import numpy as np


def cfg_tag(d):
    tag = f"{str(d['adapter'])}_{str(d['feedback'])}"
    return tag if bool(d["adapt"]) else tag + "_noadapt"


def onset_step(step_eval, bacc, frac=0.9):
    if len(bacc) == 0:
        return None
    thr = frac * float(bacc[0])
    for s, b in zip(step_eval, bacc):
        if b < thr:
            return int(s)
    return None  # never crossed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/trajectory/*.npz")
    ap.add_argument("--outdir", default="results/trajectory_csv")
    ap.add_argument("--bs", type=int, default=16, help="batch size, for the lambda_t bound")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    os.makedirs(args.outdir, exist_ok=True)

    # (which,corr,sev) -> cfgtag -> list of per-seed dicts
    traj = defaultdict(lambda: defaultdict(list))
    lam_runs = []  # balanced-population lambda_t (feedback == none)
    for f in files:
        d = np.load(f, allow_pickle=True)
        key = (str(d["which"]), str(d["corruption"]), int(d["severity"]))
        tag = cfg_tag(d)
        traj[key][tag].append(dict(step_eval=d["step_eval"], bacc=d["bacc"],
                                   lam=d["lam"], logK=float(d["logK"]), lam0=float(d["lam0"]),
                                   p=float(d["p"]), seed=int(d["seed"])))
        if str(d["feedback"]) == "none" and str(d["adapter"]).startswith("patta"):
            lam_runs.append(dict(lam=d["lam"], logK=float(d["logK"]), lam0=float(d["lam0"]),
                                 p=float(d["p"]), which=str(d["which"]),
                                 corr=str(d["corruption"]), sev=int(d["severity"])))

    summary = []
    for key in sorted(traj):
        which, corr, sev = key
        print(f"\n=== {which} / {corr} / sev{sev} ===")
        for tag in sorted(traj[key]):
            runs = traj[key][tag]
            n = min(len(r["bacc"]) for r in runs)
            if n == 0:
                continue
            steps = runs[0]["step_eval"][:n]
            B = np.stack([r["bacc"][:n] for r in runs])          # (seeds, n)
            mean, std = B.mean(0), B.std(0, ddof=1) if len(runs) > 1 else np.zeros(n)
            onsets = [onset_step(r["step_eval"], r["bacc"]) for r in runs]
            onsets_ok = [o for o in onsets if o is not None]
            onset_mean = float(np.mean(onsets_ok)) if onsets_ok else float("nan")
            print(f"  {tag:20s} init {mean[0]:.3f} -> final {mean[-1]:.3f}  "
                  f"onset~{onset_mean:.0f} (n={len(runs)}, crossed {len(onsets_ok)}/{len(runs)})")
            out = os.path.join(args.outdir, f"traj_{which}_{corr}_sev{sev}_{tag}.csv")
            with open(out, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["step", "bacc_mean", "bacc_std"])
                for s, m, sd in zip(steps, mean, std):
                    w.writerow([int(s), round(float(m), 4), round(float(sd), 4)])
            summary.append(dict(which=which, corruption=corr, severity=sev, config=tag,
                                n_seeds=len(runs), init_bacc=round(float(mean[0]), 4),
                                final_bacc=round(float(mean[-1]), 4),
                                onset_step=(round(onset_mean, 1) if onsets_ok else "")))

    # --- lambda_t distribution at K >> B (Concern 3 / Section 6.8.1) ---
    if lam_runs:
        ratios = np.concatenate([r["lam"] / max(r["lam0"], 1e-12) for r in lam_runs])
        logK = lam_runs[0]["logK"]; p = lam_runs[0]["p"]
        bound = (1.0 - math.log(args.bs) / logK) ** p
        frac_ge = float((ratios >= bound - 1e-9).mean())
        print(f"\n=== lambda_t at K>>B ({lam_runs[0]['which']}, B={args.bs}, p={p}) ===")
        print(f"  theoretical lower bound (1 - log B/log K)**p = {bound:.3f}")
        print(f"  empirical lam/lam0: min {ratios.min():.3f}  median {np.median(ratios):.3f}  "
              f"max {ratios.max():.3f}  mean {ratios.mean():.3f}")
        print(f"  fraction of steps with lam/lam0 >= bound: {frac_ge:.3f}")
        with open(os.path.join(args.outdir, "lambda_ratio_values.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["lam_over_lam0"])
            for v in ratios:
                w.writerow([round(float(v), 5)])
        hist, edges = np.histogram(ratios, bins=20, range=(0.0, 1.0))
        with open(os.path.join(args.outdir, "lambda_ratio_hist.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(hist)):
                w.writerow([round(float(edges[i]), 4), round(float(edges[i + 1]), 4), int(hist[i])])
        summary.append(dict(which=lam_runs[0]["which"], corruption="(lambda_t)", severity="",
                            config=f"bound={bound:.3f}", n_seeds=len(lam_runs),
                            init_bacc="", final_bacc="",
                            onset_step=f"frac>=bound {frac_ge:.3f}"))

    with open(os.path.join(args.outdir, "trajectory_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print("\nsaved CSVs to", args.outdir)


if __name__ == "__main__":
    main()
