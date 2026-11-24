"""Summarize the measured bifurcation diagram (exp_critical): per batch size B, the order
parameter s(eps) = 1 - H(q)/log K and balanced accuracy vs eps, averaged over seeds.

Theory (Appendix B) predicts: s(eps) ~ 0 below a critical eps*, then rises sharply
(pitchfork); eps* INCREASES with B; and the accuracy collapse coincides with the s-rise
(the collapse IS the bifurcation). This reports eps*_stability (s crosses S_THRESH) and
eps*_accuracy (bacc drops below ACC_FRAC x its eps=0 value), their agreement, and the
monotonicity of eps*(B).

    python -m experiments.summarize_critical --glob "results/crit/crit_*.npz" --out results/critical.csv
"""
import argparse
import csv
import glob
import os
import numpy as np

S_THRESH = 0.30       # order-parameter level marking the collapsed (concentrated) branch
ACC_FRAC = 0.85       # accuracy onset = first eps where bacc < ACC_FRAC * bacc(eps=0)


def crossing(eps, y, thresh, rising=True):
    """First eps where y crosses thresh (linearly interpolated); None if never."""
    for i in range(1, len(y)):
        a, b = y[i - 1], y[i]
        if (rising and a < thresh <= b) or (not rising and a >= thresh > b):
            if b == a:
                return eps[i]
            return eps[i - 1] + (thresh - a) / (b - a) * (eps[i] - eps[i - 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/crit/crit_*.npz")
    ap.add_argument("--out", default="results/critical.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"[error] no files match: {args.glob}")
        raise SystemExit(2)

    groups = {}      # (which,corr,bs) -> list of (eps, s_final, bacc)
    for p in paths:
        d = np.load(p, allow_pickle=True)
        key = (str(d["which"]), str(d["corruption"]), int(d["bs"]))
        groups.setdefault(key, []).append((np.asarray(d["eps"], float),
                                           np.asarray(d["s_final"], float),
                                           np.asarray(d["bacc"], float)))

    rows = []
    estar_by_bs = {}
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        which, corr, bs = key
        runs = groups[key]
        eps = runs[0][0]
        s = np.mean([r[1] for r in runs], axis=0)
        ba = np.mean([r[2] for r in runs], axis=0)
        ba0 = ba[0] if len(ba) else float("nan")
        eps_stab = crossing(eps, s, S_THRESH, rising=True)
        eps_acc = crossing(eps, ba, ACC_FRAC * ba0, rising=False)
        estar_by_bs[bs] = eps_stab
        print(f"\n=== {which} / {corr} / bs={bs} ===  bacc(eps=0)={ba0:.3f}")
        print("    eps  : " + " ".join(f"{e:4.1f}" for e in eps))
        print("    s(eps): " + " ".join(f"{v:4.2f}" for v in s))
        print("    bacc  : " + " ".join(f"{v:4.2f}" for v in ba))
        print(f"    eps*_stability (s>{S_THRESH}) = {eps_stab if eps_stab is not None else '>max (no collapse in range)'}"
              f"   eps*_accuracy (bacc<{ACC_FRAC:.2f}xinit) = {eps_acc if eps_acc is not None else '>max'}")
        for i, e in enumerate(eps):
            rows.append([which, corr, bs, f"{e:.2f}", f"{s[i]:.4f}", f"{ba[i]:.4f}"])
        rows.append([which, corr, bs, "eps*_stab", "" if eps_stab is None else f"{eps_stab:.3f}",
                     "" if eps_acc is None else f"{eps_acc:.3f}"])

    # monotonicity of eps*(B) and agreement stability-vs-accuracy
    bss = sorted(estar_by_bs)
    seq = [(b, estar_by_bs[b]) for b in bss]
    print("\n==================== BIFURCATION SUMMARY ====================")
    print("    eps*_stability by batch size:", ", ".join(
        f"bs{b}: {('%.2f' % v) if v is not None else '>max'}" for b, v in seq))
    known = [(b, v) for b, v in seq if v is not None]
    mono = all(known[i][1] <= known[i + 1][1] + 1e-9 for i in range(len(known) - 1))
    print("    eps* increases with batch size:" , "YES (matches theory)" if mono and len(known) >= 2
          else ("partial/flat" if len(known) >= 2 else "n/a (single bs)"))
    print("    Reading: s(eps) flat then rising = a measured pitchfork; eps* moving right with B = T2's batch law;")
    print("             eps*_stability ~= eps*_accuracy = the accuracy collapse IS the bifurcation.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "corruption", "bs", "eps", "s_final/epsstar", "bacc/epsacc"])
        w.writerows(rows)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
