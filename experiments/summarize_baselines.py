"""Baseline comparison table: per (dataset, corruption, batch), each adapter's
balanced accuracy averaged over seeds, and whether PATTA beats the baselines.

Decides the paper's framing: if PATTA tops the anti-collapse family -> method+
theory paper; if it merely ties -> theory/diagnosis paper (framing is the win).

    python -m experiments.summarize_baselines --glob "results/base/base_*.npz" --out results/baselines.csv
"""
import argparse
import csv
import glob
import os
import numpy as np

PRIMARY = "patta"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/base/base_*.npz")
    ap.add_argument("--out", default="results/baselines.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"[error] no files match: {args.glob}")
        raise SystemExit(2)

    # groups[(which, corr, bs)][adapter] = list of (bacc, sacc) across seeds
    groups = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        key = (str(d["which"]), str(d["corruption"]), int(d["bs"]))
        names = [str(a) for a in d["adapters"]]
        g = groups.setdefault(key, {})
        for i, a in enumerate(names):
            g.setdefault(a, []).append((float(d["final_bacc"][i]), float(d["final_acc"][i])))

    rows = []
    settings_patta_best, settings_total = 0, 0
    margins_vs_best = []
    for key in sorted(groups):
        which, corr, bs = key
        g = groups[key]
        means = {a: (np.mean([v[0] for v in lst]), np.std([v[0] for v in lst]),
                     np.mean([v[1] for v in lst])) for a, lst in g.items()}
        order = sorted(means, key=lambda a: means[a][0], reverse=True)
        print(f"\n=== {which} / {corr} / bs={bs} ===  (balanced acc, mean+/-std over seeds)")
        for a in order:
            mb, sb, ms = means[a]
            star = "  <-- ours" if a == PRIMARY else ""
            print(f"    {a:10s} bal={mb:.3f}+/-{sb:.3f}  stream={ms:.3f}{star}")
            rows.append([which, corr, bs, a, f"{mb:.4f}", f"{sb:.4f}", f"{ms:.4f}"])

        # PATTA vs best non-source, non-patta baseline
        baselines = [a for a in means if a not in (PRIMARY, "source")]
        if PRIMARY in means and baselines:
            settings_total += 1
            best_base = max(baselines, key=lambda a: means[a][0])
            margin = means[PRIMARY][0] - means[best_base][0]
            margins_vs_best.append(margin)
            tops = means[PRIMARY][0] >= means[best_base][0] - 1e-9
            settings_patta_best += int(tops)
            print(f"    -> PATTA {means[PRIMARY][0]:.3f} vs best baseline "
                  f"{best_base} {means[best_base][0]:.3f}  (margin {margin:+.3f}, "
                  f"{'PATTA tops' if tops else 'baseline tops'})")

    if settings_total:
        mv = np.array(margins_vs_best)
        print(f"\nOVERALL: PATTA >= best baseline in {settings_patta_best}/{settings_total} settings;  "
              f"mean margin vs best baseline {mv.mean():+.3f} +/- {mv.std():.3f}")
        if settings_patta_best == settings_total and mv.mean() > 0.02:
            verdict = "METHOD-FIRST (PATTA beats the anti-collapse family)"
        elif settings_patta_best >= 0.6 * settings_total:
            verdict = "METHOD COMPETITIVE (PATTA >= baselines mostly; emphasize theory + the derived-vs-heuristic angle)"
        else:
            verdict = "THEORY-FIRST (PATTA ties/loses on method; sell framing, bifurcation, hidden-metric)"
        print("FRAMING:", verdict)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "corruption", "bs", "adapter", "bal_mean", "bal_std", "stream_mean"])
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
