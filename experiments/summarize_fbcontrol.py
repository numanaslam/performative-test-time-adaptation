"""Aggregate the feedback-generator control runs (exp_fbcontrol) across seeds.

For each (which, corruption, severity, mode) prints the seed-mean initial and final
balanced accuracy and the final induced-prior entropy, and writes a CSV. The frozen
mode staying flat (final ~ init) while its prior entropy collapses shows the balanced-
accuracy collapse is adaptation-driven; adapt_oracle staying far above adapt_pred shows
it needs prediction-dependent feedback.

    python -m experiments.summarize_fbcontrol --glob "results/fbcontrol/*.npz" --out results/fbcontrol.csv
"""
import argparse
import csv
import glob
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/fbcontrol/*.npz")
    ap.add_argument("--out", default="results/fbcontrol.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")

    agg = defaultdict(lambda: defaultdict(list))   # (which,corr,sev) -> mode -> list of (init,final,priorH,logK)
    for f in files:
        d = np.load(f, allow_pickle=True)
        key = (str(d["which"]), str(d["corruption"]), int(d["severity"]))
        modes = [str(m) for m in d["modes"]]
        for i, m in enumerate(modes):
            agg[key][m].append((float(d["init_bacc"][i]), float(d["final_bacc"][i]),
                                float(d["final_prior_entropy"][i]), float(d["logK"])))

    rows = []
    for key in sorted(agg):
        which, corr, sev = key
        print(f"\n=== {which} / {corr} / sev{sev} ===")
        for mode in ["adapt_pred", "frozen_pred", "adapt_oracle", "frozen_oracle"]:
            if mode not in agg[key]:
                continue
            a = np.array(agg[key][mode])
            n = len(a)

            def ms(col):
                m = float(a[:, col].mean())
                s = float(a[:, col].std(ddof=1)) if n > 1 else 0.0
                return m, s

            init, init_sd = ms(0)
            fin, fin_sd = ms(1)
            pH, pH_sd = ms(2)
            logK = float(a[:, 3].mean())
            print(f"  {mode:13s} bal {init:.3f}+/-{init_sd:.3f} -> {fin:.3f}+/-{fin_sd:.3f}  "
                  f"prior_H {pH:.2f}+/-{pH_sd:.2f} /{logK:.2f}  (n={n})")
            rows.append(dict(which=which, corruption=corr, severity=sev, mode=mode, n=n,
                             init_bacc=round(init, 4), init_sd=round(init_sd, 4),
                             final_bacc=round(fin, 4), final_sd=round(fin_sd, 4),
                             final_prior_entropy=round(pH, 4), prior_entropy_sd=round(pH_sd, 4),
                             logK=round(logK, 4)))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
