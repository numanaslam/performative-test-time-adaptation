"""Lambda-SCHEDULE comparison: can a smarter marginal-entropy *schedule* reclaim the
CIFAR-100 near-total-collapse regime (where plain PATTA under-corrected and the fixed
`balance` heuristic won) WITHOUT losing PATTA's CIFAR-10 advantage or its benign-regime
inertness?

All variants share the same MAX strength (lam0=1.0) as `balance`; only the schedule
SHAPE differs (u = (Hmax-H(q))/Hmax is the measured loop pressure):
  patta        reactive linear ramp   lam = u            (p=1,  floor=0)   [baseline]
  patta_steep  concave ramp           lam = u^0.5        (p=.5, floor=0)   [inert when uniform]
  patta_floor  always-on floor        lam = max(.1, u)   (p=1,  floor=.1)  [preventive]
  patta_sev    steep + floor          lam = max(.1,u^.5) (p=.5, floor=.1)  [the proposal]
  balance      fixed                  lam = 1            (reference to match on CIFAR-100)

    python -m experiments.summarize_lambda --glob "results/lambda/lam_*.npz" \
        --inert_glob "results/lambda_inert/*.npz" --out results/lambda.csv
"""
import argparse
import csv
import glob
import os
import numpy as np

VARIANTS = ["patta", "patta_steep", "patta_floor", "patta_sev"]
REF = "balance"          # fixed-weight heuristic the variants must match on CIFAR-100
NOISE = 0.01             # within-noise tolerance for "matches"


def load_groups(pattern):
    """groups[(which, corr, bs)][adapter] = list of balanced acc across seeds."""
    groups = {}
    for p in sorted(glob.glob(pattern)):
        d = np.load(p, allow_pickle=True)
        key = (str(d["which"]), str(d["corruption"]), int(d["bs"]))
        names = [str(a) for a in d["adapters"]]
        g = groups.setdefault(key, {})
        for i, a in enumerate(names):
            g.setdefault(a, []).append((float(d["final_bacc"][i]), float(d["final_acc"][i])))
    return groups


def mean_bal(group):
    return {a: float(np.mean([v[0] for v in lst])) for a, lst in group.items()}


def mean_stream(group):
    return {a: float(np.mean([v[1] for v in lst])) for a, lst in group.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/lambda/lam_*.npz")
    ap.add_argument("--inert_glob", default="results/lambda_inert/*.npz")
    ap.add_argument("--out", default="results/lambda.csv")
    args = ap.parse_args()

    groups = load_groups(args.glob)
    if not groups:
        print(f"[error] no files match: {args.glob}")
        raise SystemExit(2)

    rows = []
    # per-cell balanced acc, split by dataset
    cell_bal = {}    # key -> {adapter: mean bal}
    for key in sorted(groups):
        which, corr, bs = key
        mb = mean_bal(groups[key])
        ms = mean_stream(groups[key])
        cell_bal[key] = mb
        order = sorted(mb, key=lambda a: mb[a], reverse=True)
        print(f"\n=== {which} / {corr} / bs={bs} ===  (balanced acc, mean over seeds)")
        for a in order:
            tag = "  <-- proposal" if a == "patta_sev" else ("  <-- baseline PATTA" if a == "patta" else "")
            print(f"    {a:12s} bal={mb[a]:.3f}  stream={ms[a]:.3f}{tag}")
            rows.append([which, corr, bs, a, f"{mb[a]:.4f}", f"{ms[a]:.4f}"])

    c10_cells = [k for k in cell_bal if k[0] == "cifar10"]
    c100_cells = [k for k in cell_bal if k[0] == "cifar100"]

    # benign-regime inertness: |variant - tent| in the no-collapse cell(s)
    inert = load_groups(args.inert_glob)
    benign_tax = {v: None for v in VARIANTS}
    if inert:
        taxes = {v: [] for v in VARIANTS}
        print("\n=== benign-regime inertness (no collapse: variant should ~= tent) ===")
        for key in sorted(inert):
            mb = mean_bal(inert[key])
            t = mb.get("tent", float("nan"))
            line = f"    {key[0]}/{key[1]}/bs{key[2]}:  tent={t:.3f}"
            for v in VARIANTS:
                if v in mb:
                    line += f"  {v}={mb[v]:.3f}"
                    taxes[v].append(t - mb[v])   # +ve => variant worse than tent (a tax)
            print(line)
        benign_tax = {v: (float(np.mean(taxes[v])) if taxes[v] else None) for v in VARIANTS}

    # per-variant summary vs the fixed `balance` reference and vs tent
    print("\n=== variant summary (vs fixed `balance`, split by dataset) ===")
    print(f"    {'variant':12s} {'c10 bal':>8s} {'c10 vs bal':>11s} "
          f"{'c100 bal':>9s} {'c100 vs bal':>12s} {'c100 won/N':>11s} {'benign tax':>11s}")
    summary = {}
    for v in VARIANTS:
        c10_m = [cell_bal[k][v] - cell_bal[k].get(REF, np.nan) for k in c10_cells if v in cell_bal[k]]
        c100_m = [cell_bal[k][v] - cell_bal[k].get(REF, np.nan) for k in c100_cells if v in cell_bal[k]]
        c10_bal = np.mean([cell_bal[k][v] for k in c10_cells if v in cell_bal[k]]) if c10_cells else np.nan
        c100_bal = np.mean([cell_bal[k][v] for k in c100_cells if v in cell_bal[k]]) if c100_cells else np.nan
        c100_won = sum(1 for m in c100_m if m >= -NOISE)
        tax = benign_tax.get(v)
        tax_s = f"{tax:+.3f}" if tax is not None else "  n/a"
        summary[v] = dict(c10_bal=c10_bal, c10_m=np.mean(c10_m) if c10_m else np.nan,
                          c100_bal=c100_bal, c100_m=np.mean(c100_m) if c100_m else np.nan,
                          c100_won=c100_won, c100_N=len(c100_m), tax=tax)
        print(f"    {v:12s} {c10_bal:8.3f} {np.mean(c10_m) if c10_m else np.nan:+11.3f} "
              f"{c100_bal:9.3f} {np.mean(c100_m) if c100_m else np.nan:+12.3f} "
              f"{c100_won:>6d}/{len(c100_m):<4d} {tax_s:>11s}")

    # recommendation
    def reclaims(v):      # matches/beats fixed balance on every CIFAR-100 cell (within noise)
        return summary[v]["c100_N"] > 0 and summary[v]["c100_won"] == summary[v]["c100_N"]

    def keeps_c10(v):     # does not regress below baseline PATTA on CIFAR-10
        base = summary["patta"]["c10_bal"]
        return not np.isnan(summary[v]["c10_bal"]) and summary[v]["c10_bal"] >= base - NOISE

    def safe(v):          # benign tax small (inertness approx preserved)
        t = summary[v]["tax"]
        return t is None or t <= 0.02

    print("\n=== RECOMMENDATION ===")
    # prefer the most inert variant that reclaims c100 and keeps c10
    picked = None
    for v in ["patta_steep", "patta_sev", "patta_floor"]:   # inert-first preference order
        if reclaims(v) and keeps_c10(v):
            if picked is None or (safe(v) and not safe(picked)):
                picked = v
    if picked:
        s = summary[picked]
        tag = "keeps full inertness" if safe(picked) else f"costs {s['tax']:+.3f} benign tax"
        print(f"  ADOPT `{picked}`: reclaims CIFAR-100 ({s['c100_won']}/{s['c100_N']} cells >= balance, "
              f"mean {s['c100_m']:+.3f}), keeps CIFAR-10 (mean vs balance {s['c10_m']:+.3f}), {tag}.")
        print("  -> PATTA competitive with the anti-collapse family everywhere; method claim strengthened.")
    else:
        best = max(VARIANTS, key=lambda v: summary[v]["c100_won"])
        s = summary[best]
        print(f"  NO variant fully reclaims CIFAR-100 while keeping CIFAR-10. Best is `{best}` "
              f"({s['c100_won']}/{s['c100_N']} c100 cells >= balance, c10 vs balance {s['c10_m']:+.3f}, "
              f"tax {('%+.3f' % s['tax']) if s['tax'] is not None else 'n/a'}).")
        print("  -> keep the THEORY-FIRST framing; report the schedule study as an ablation, not a new SOTA.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "corruption", "bs", "adapter", "bal_mean", "stream_mean"])
        w.writerows(rows)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
