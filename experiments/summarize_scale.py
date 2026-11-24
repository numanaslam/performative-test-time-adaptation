"""Scale sweep summary (severity x eps): at ImageNet scale, is the collapse *performative*
(isolated by comparing eps=0 vs eps=1), is it hidden by stream accuracy, and does PATTA
rescue it — in a regime where the model STARTS healthy (lower severity), not at the sev5
floor where ResNet-50 is already ~dead?

Groups by (dataset, corruption, severity, bs). Within each, compares eps=0 (reference) to
eps=1 (performative). Diagnostics per cell:
  - start health   = source (frozen) balanced acc
  - performative harm = tent balanced at eps=0  ->  at eps=1  (the isolated feedback effect)
  - hidden metric  = tent stream - tent balanced at eps=1
  - rescue         = best PATTA balanced vs tent, at eps=1

    python -m experiments.summarize_scale --glob "results/scale/scale_in_*_sev*.npz" --out results/scale.csv
"""
import argparse
import csv
import glob
import os
import numpy as np

PATTAS = ["patta_sev", "patta"]
HEALTHY = 0.15
HIDDEN = 0.10
MARGIN = 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/scale/scale_in_*_sev*.npz")
    ap.add_argument("--out", default="results/scale.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"[error] no files match: {args.glob}")
        raise SystemExit(2)

    # groups[(which,corr,sev,bs)][eps][adapter] = list of (bal, stream)
    groups = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        key = (str(d["which"]), str(d["corruption"]), int(d["severity"]), int(d["bs"]))
        eps = round(float(d["eps"]), 3)
        names = [str(a) for a in d["adapters"]]
        g = groups.setdefault(key, {}).setdefault(eps, {})
        for i, a in enumerate(names):
            g.setdefault(a, []).append((float(d["final_bacc"][i]), float(d["final_acc"][i])))

    rows = []
    for key in sorted(groups):
        which, corr, sev, bs = key
        byeps = groups[key]
        print(f"\n==================== {which} / {corr} / sev={sev} / bs={bs} ====================")

        def mean(eps, a, j):
            lst = byeps.get(eps, {}).get(a)
            return float(np.mean([v[j] for v in lst])) if lst else float("nan")

        for eps in sorted(byeps):
            mb = {a: mean(eps, a, 0) for a in byeps[eps]}
            ms = {a: mean(eps, a, 1) for a in byeps[eps]}
            order = sorted(mb, key=lambda a: mb[a], reverse=True)
            print(f"  -- eps={eps} --   (balanced | stream)")
            for a in order:
                tag = "  <-- ours" if a in ("patta", "patta_sev") else ("  (frozen)" if a == "source" else "")
                print(f"     {a:11s} bal={mb[a]:.3f}  stream={ms[a]:.3f}{tag}")
                rows.append([which, corr, sev, bs, eps, a, f"{mb[a]:.4f}", f"{ms[a]:.4f}"])

        # diagnostics (need eps=1; eps=0 optional reference)
        src = mean(1.0, "source", 0)
        t0, t1 = mean(0.0, "tent", 0), mean(1.0, "tent", 0)
        t1s = mean(1.0, "tent", 1)
        best_p = max((mean(1.0, a, 0) for a in PATTAS), default=float("nan"))
        best_p_name = max((a for a in PATTAS if not np.isnan(mean(1.0, a, 0))),
                          key=lambda a: mean(1.0, a, 0), default="-")
        healthy = src >= HEALTHY
        perf_harm = (t0 - t1) if not np.isnan(t0) else float("nan")  # performative drop in TENT
        performative = (not np.isnan(t0)) and (perf_harm >= MARGIN)   # eps=0 healthy -> eps=1 collapsed
        hidden_gap = t1s - t1
        hidden = hidden_gap >= HIDDEN
        mitigates = healthy and (best_p >= t1 + MARGIN)               # beats collapsed TENT
        restores = mitigates and (best_p >= src - 0.02)               # recovers to >= not-adapting (frozen)

        print(f"  >> start(source,eps1)={src:.3f}  "
              f"performative harm(tent eps0->eps1)={'%.3f->%.3f (%+.3f)'%(t0,t1,-perf_harm) if not np.isnan(t0) else 'n/a (no eps0)'}")
        print(f"     hidden metric(eps1 tent): stream {t1s:.3f} vs balanced {t1:.3f} (gap {hidden_gap:+.3f})")
        print(f"     rescue(eps1): {best_p_name} {best_p:.3f} vs tent {t1:.3f} (margin {best_p-t1:+.3f}); "
              f"vs frozen source {src:.3f}")
        if not healthy:
            verdict = "FLOOR (source ~dead) -- rescue untestable; " + ("hidden-metric SHOWN" if hidden else "weak hidden-metric")
        elif performative and restores:
            verdict = "PERFORMATIVE COLLAPSE + PATTA RESTORES (>= frozen) at scale" + ("; hidden-metric SHOWN" if hidden else "")
        elif performative:
            verdict = (f"PERFORMATIVE COLLAPSE + hidden-metric SHOWN; PATTA only MITIGATES vs TENT "
                       f"({best_p:.3f} vs {t1:.3f}) and does NOT restore -- frozen source ({src:.3f}) beats all adapters")
        elif mitigates and hidden:
            verdict = "collapse + hidden-metric SHOWN, but eps0 shows little performative harm (TENT collapses already at eps=0); PATTA only mitigates"
        elif hidden:
            verdict = "collapse + hidden-metric SHOWN; PATTA does NOT recover at this severity"
        else:
            verdict = "no clear collapse/rescue"
        print(f"  >> VERDICT: {verdict}")
        rows.append([which, corr, sev, bs, "", "_verdict", verdict, ""])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "corruption", "severity", "bs", "eps", "adapter", "bal_mean/verdict", "stream_mean"])
        w.writerows(rows)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
