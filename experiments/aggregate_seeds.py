"""Average multi-seed probe/rescue runs into mean +/- std summaries.

Probe: stacks the (eps x bs) grids (bacc, H(q), stream acc, ttc) across seeds
and writes a mean grid whose keys match a single probe npz, so plot_phase.py
can render it unchanged -- plus *_std grids for error reporting.

Rescue: gathers the per-seed scalar verdict (tent vs patta balanced accuracy at
the collapse cell) and reports mean +/- std and the win count across seeds.

    python -m experiments.aggregate_seeds --kind probe  --glob "results/seeds/probe_tent_s*.npz" --out results/probe_tent_mean.npz
    python -m experiments.aggregate_seeds --kind rescue --glob "results/seeds/rescue_s*.npz"      --out results/rescue_mean.npz
"""
import argparse
import csv
import glob
import math
import os
import numpy as np


def _load(paths):
    return [(p, np.load(p)) for p in paths]


def aggregate_probe(paths, out, which):
    data = _load(paths)
    K = 100 if which == "cifar100" else 10
    logK = math.log(K)
    eps = data[0][1]["eps"]
    bs = data[0][1]["bs"]
    keys = ["bacc", "Hq", "acc", "ttc"]
    stacks = {k: np.stack([d[k].astype(float) for _, d in data]) for k in keys}
    means = {k: np.nanmean(v, axis=0) for k, v in stacks.items()}
    stds = {k: np.nanstd(v, axis=0) for k, v in stacks.items()}

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, eps=eps, bs=bs, n_seeds=len(data),
             bacc=means["bacc"], Hq=means["Hq"], acc=means["acc"], ttc=means["ttc"],
             bacc_std=stds["bacc"], Hq_std=stds["Hq"], acc_std=stds["acc"], ttc_std=stds["ttc"])
    print(f"[probe] {len(data)} seeds -> {out}")
    for p, _ in data:
        print("   ", os.path.basename(p))

    # headline: balanced-accuracy drop with eps at each batch (mean +/- std)
    print(f"    balanced accuracy, mean+/-std over {len(data)} seeds (rows=eps, cols=bs):")
    header = "     eps\\bs " + "".join(f"{int(b):>8d}" for b in bs)
    print(header)
    for i, e in enumerate(eps):
        row = "".join(f" {means['bacc'][i, j]:.3f}" for j in range(len(bs)))
        print(f"      {e:4.1f}  {row}")
    # the harm: bacc at eps=0 vs eps=max, per batch
    print("    harm (bacc at eps_min -> eps_max), per batch:")
    for j, b in enumerate(bs):
        lo, hi = means["bacc"][0, j], means["bacc"][-1, j]
        s = stds["bacc"][-1, j]
        print(f"      bs={int(b):4d}:  {lo:.3f} -> {hi:.3f}  (delta {hi - lo:+.3f}, std {s:.3f})")


def aggregate_rescue(paths, out):
    data = _load(paths)

    def col(key):
        return np.array([float(d[key]) for _, d in data])

    tb, pb = col("tent_final_bacc"), col("patta_final_bacc")
    ts, ps = col("tent_final_acc"), col("patta_final_acc")
    th, ph = col("tent_final_Hq"), col("patta_final_Hq")
    wins = int(np.sum(pb >= tb))
    margin = pb - tb

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, n_seeds=len(data),
             tent_final_bacc=tb, patta_final_bacc=pb,
             tent_final_acc=ts, patta_final_acc=ps,
             tent_final_Hq=th, patta_final_Hq=ph, margin=margin)
    csvp = os.path.splitext(out)[0] + ".csv"
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "tent_bal", "patta_bal", "margin", "tent_stream", "patta_stream"])
        for (p, _), a, b, m, c, d2 in zip(data, tb, pb, margin, ts, ps):
            w.writerow([os.path.basename(p), f"{a:.4f}", f"{b:.4f}", f"{m:.4f}", f"{c:.4f}", f"{d2:.4f}"])

    print(f"[rescue] {len(data)} seeds -> {out}, {csvp}")
    print(f"    balanced acc:  TENT {tb.mean():.3f}+/-{tb.std():.3f}   "
          f"PATTA {pb.mean():.3f}+/-{pb.std():.3f}")
    print(f"    stream   acc:  TENT {ts.mean():.3f}+/-{ts.std():.3f}   "
          f"PATTA {ps.mean():.3f}+/-{ps.std():.3f}")
    print(f"    H(q):          TENT {th.mean():.3f}+/-{th.std():.3f}   "
          f"PATTA {ph.mean():.3f}+/-{ph.std():.3f}")
    print(f"    PATTA wins balanced acc in {wins}/{len(data)} seeds; "
          f"mean margin {margin.mean():+.3f} +/- {margin.std():.3f}")
    print("    VERDICT:", "ROBUST PASS" if wins == len(data) and margin.mean() > 0.05
          else ("MAJORITY PASS" if wins > len(data) / 2 else "NOT ROBUST"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["probe", "rescue"])
    ap.add_argument("--glob", required=True, help="quoted glob, e.g. results/seeds/probe_tent_s*.npz")
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"[error] no files match: {args.glob}")
        raise SystemExit(2)
    if args.kind == "probe":
        aggregate_probe(paths, args.out, args.which)
    else:
        aggregate_rescue(paths, args.out)


if __name__ == "__main__":
    main()
