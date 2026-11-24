"""Summarize the endogenous-feedback experiment: on a BALANCED IID stream with no eps and
no exogenous correlation, does the model's confidence-gated replay buffer self-induce class
imbalance and collapse (alpha>0) while a no-feedback control (alpha=0) stays healthy?

Reads results/endo/*.npz, groups by (which, corruption), reports each adapter's balanced
acc, stream acc, and u_buf (buffer-marginal concentration) across alpha. Verdict:
  (1) control alpha=0 stays healthy (balanced ~ source-adapted baseline),
  (2) full feedback alpha=1 collapses balanced acc with u_buf high -> PURELY ENDOGENOUS
      (no eps, no correlation: the model manufactured the imbalance on balanced data),
  (3) PATTA recovers on the same loop.

    python -m experiments.summarize_endogenous --glob "results/endo/endo_*.npz" --out results/endo.csv
"""
import argparse
import csv
import glob
import os
import numpy as np

COLLAPSE_ADAPTER = "tent"
FIX = ["patta_sev", "patta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/endo/endo_*.npz")
    ap.add_argument("--out", default="results/endo.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"[error] no files match: {args.glob}")
        raise SystemExit(2)

    groups = {}      # (which,corr) -> alpha -> adapter -> list of (bacc,sacc,ubuf)
    for p in paths:
        d = np.load(p, allow_pickle=True)
        key = (str(d["which"]), str(d["corruption"]))
        a = round(float(d["alpha"]), 3)
        names = [str(x) for x in d["adapters"]]
        g = groups.setdefault(key, {}).setdefault(a, {})
        for i, nm in enumerate(names):
            g.setdefault(nm, []).append((float(d["final_bacc"][i]), float(d["final_acc"][i]),
                                         float(d["u_buf"][i])))

    rows = []
    for key in sorted(groups):
        which, corr = key
        alphas = sorted(groups[key])
        a_lo, a_hi = alphas[0], alphas[-1]

        def mean(a, nm, j):
            lst = groups[key].get(a, {}).get(nm)
            return float(np.mean([v[j] for v in lst])) if lst else float("nan")

        print(f"\n==================== {which} / {corr} ====================")
        for a in alphas:
            print(f"  -- alpha={a} (0=no feedback control, 1=full self-selected buffer) --   bal / stream / u_buf")
            for nm in sorted(groups[key][a]):
                b, s, u = mean(a, nm, 0), mean(a, nm, 1), mean(a, nm, 2)
                print(f"     {nm:11s} bal={b:.3f}  stream={s:.3f}  u_buf={u:.3f}")
                rows.append([which, corr, a, nm, f"{b:.4f}", f"{s:.4f}", f"{u:.4f}"])

        if a_lo == a_hi:
            continue
        c_lo = mean(a_lo, COLLAPSE_ADAPTER, 0)
        c_hi = mean(a_hi, COLLAPSE_ADAPTER, 0)
        u_hi = mean(a_hi, COLLAPSE_ADAPTER, 2)
        fix = next((f for f in FIX if not np.isnan(mean(a_hi, f, 0))), None)
        f_hi = mean(a_hi, fix, 0) if fix else float("nan")
        print(f"\n  attribution (balanced iid, NO eps, NO correlation):")
        print(f"    control alpha=0 ({COLLAPSE_ADAPTER}):  balanced {c_lo:.3f}  (healthy = no self-feedback)")
        print(f"    full feedback alpha={a_hi} ({COLLAPSE_ADAPTER}):  balanced {c_hi:.3f}, u_buf {u_hi:.3f}")
        if fix:
            print(f"    fix ({fix}) alpha={a_hi}:  balanced {c_hi:.3f} -> {f_hi:.3f}  (+{f_hi-c_hi:.3f})")

        endogenous = (c_hi < c_lo - 0.10) and (u_hi > 0.25)
        recovers = fix is not None and f_hi > c_hi + 0.03
        print("\n  VERDICT:",
              "ENDOGENOUS COLLAPSE CONFIRMED (model self-induced imbalance on balanced iid, no eps/correlation)"
              if endogenous else
              "no clean endogenous collapse (self-feedback channel too weak at these settings; "
              "try higher alpha / larger buffer / longer horizon)",
              "+ PATTA recovers" if (endogenous and recovers) else "")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "corruption", "alpha", "adapter", "bal_mean", "stream_mean", "u_buf_mean"])
        w.writerows(rows)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
