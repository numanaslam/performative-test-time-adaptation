"""Summarize the NATURAL-feedback experiment: does performative collapse arise with NO
injected eps -- purely from a correlated stream + the adapter's own confidence selection?

Reads results/natural/*.npz, groups by (which, corruption), and for the lowest vs highest
stream-correlation rho reports each adapter's balanced acc, stream acc, and natural
performativity pressure u_sel. Verdict logic:
  (1) correlation+selection collapses balanced acc (selection adapter at rho_hi << at rho_lo),
  (2) stream acc hides it (stream >> balanced at rho_hi) -- the hidden metric, with no eps,
  (3) u_sel is high at rho_hi (the natural induced marginal is concentrated),
  (4) PATTA recovers balanced acc on the same natural loop.

    python -m experiments.summarize_natural --glob "results/natural/nat_*.npz" --out results/natural.csv
"""
import argparse
import csv
import glob
import os
import numpy as np

SELECT = ["eata_lite", "sar"]        # adapters whose confidence selection IS the natural channel
FIX = ["patta", "patta_sev", "balance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/natural/nat_*.npz")
    ap.add_argument("--out", default="results/natural.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"[error] no files match: {args.glob}")
        raise SystemExit(2)

    # groups[(which,corr)][rho][adapter] = list of (bacc, sacc, usel)
    groups = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        key = (str(d["which"]), str(d["corruption"]))
        rho = round(float(d["rho"]), 3)
        names = [str(a) for a in d["adapters"]]
        g = groups.setdefault(key, {}).setdefault(rho, {})
        for i, a in enumerate(names):
            g.setdefault(a, []).append((float(d["final_bacc"][i]), float(d["final_acc"][i]),
                                        float(d["u_sel"][i])))

    rows = []
    for key in sorted(groups):
        which, corr = key
        rhos = sorted(groups[key])
        rho_lo, rho_hi = rhos[0], rhos[-1]

        def mean(rho, a, j):
            lst = groups[key].get(rho, {}).get(a)
            return float(np.mean([v[j] for v in lst])) if lst else float("nan")

        print(f"\n==================== {which} / {corr} ====================")
        for rho in rhos:
            print(f"  -- rho={rho} (0=iid, ->1=correlated) --   bal / stream / u_sel")
            for a in sorted(groups[key][rho]):
                b, s, u = mean(rho, a, 0), mean(rho, a, 1), mean(rho, a, 2)
                print(f"     {a:11s} bal={b:.3f}  stream={s:.3f}  u_sel={u:.3f}")
                rows.append([which, corr, rho, a, f"{b:.4f}", f"{s:.4f}", f"{u:.4f}"])

        # attribution at rho_hi using whichever selection adapter is present
        sel = next((a for a in SELECT if a in groups[key].get(rho_hi, {})), None)
        fix = next((a for a in FIX if a in groups[key].get(rho_hi, {})), None)
        if sel and rho_lo != rho_hi:
            b_lo = mean(rho_lo, sel, 0); b_hi = mean(rho_hi, sel, 0)
            s_hi = mean(rho_hi, sel, 1); u_hi = mean(rho_hi, sel, 2)
            t_lo = mean(rho_lo, "tent", 0); t_hi = mean(rho_hi, "tent", 0)
            print(f"\n  attribution (no eps injected):")
            print(f"    correlation alone (tent):   bal {t_lo:.3f} (iid) -> {t_hi:.3f} (corr)")
            print(f"    + selection ({sel}):   bal {b_lo:.3f} (iid) -> {b_hi:.3f} (corr)  "
                  f"[selection is the natural performative channel]")
            print(f"    hidden metric @rho_hi:      stream {s_hi:.3f}  vs  balanced {b_hi:.3f}  "
                  f"(gap {s_hi-b_hi:+.3f})")
            print(f"    natural pressure @rho_hi:   u_sel {u_hi:.3f}   (eps-free performativity)")
            if fix:
                f_hi = mean(rho_hi, fix, 0)
                print(f"    fix ({fix}) @rho_hi:     bal {b_hi:.3f} -> {f_hi:.3f}  (margin {f_hi-b_hi:+.3f})")

            collapse = (b_hi < b_lo - 0.10) and (u_hi > 0.25)      # balanced craters + selected marginal concentrates
            hidden = s_hi > b_hi + 0.05                             # does stream MASK the collapse?
            sel_effect = b_hi < t_hi - 0.02                         # does selection worsen it vs adapt-on-all?
            f_hi = mean(rho_hi, fix, 0) if fix else float("nan")
            recovers = fix is not None and f_hi > b_hi + 0.03
            print("\n  VERDICT (eps-free, correlated stream):")
            print("   - collapse:",
                  f"CONFIRMED ({sel} balanced {b_lo:.3f} -> {b_hi:.3f} as rho 0 -> {rho_hi}; u_sel {u_hi:.3f})"
                  if collapse else "not clearly observed")
            print("   - hidden metric:",
                  f"present (stream {s_hi:.3f} > balanced {b_hi:.3f})" if hidden else
                  f"ABSENT (stream {s_hi:.3f} <= balanced {b_hi:.3f}) -- stream-masking is specific to the SELF-INDUCED eps-loop, "
                  f"not the exogenous correlated stream")
            print("   - channel:",
                  f"selection worsens it ({sel} {b_hi:.3f} < tent {t_hi:.3f})" if sel_effect else
                  f"CORRELATION-DOMINATED: tent (no selection) also collapses to {t_hi:.3f}, so the endogenous "
                  f"selection channel is NOT cleanly isolated here")
            print("   - fix:",
                  f"PATTA recovers balanced {b_hi:.3f} -> {f_hi:.3f} (+{f_hi-b_hi:.3f})" if recovers else "no recovery")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "corruption", "rho", "adapter", "bal_mean", "stream_mean", "u_sel_mean"])
        w.writerows(rows)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
