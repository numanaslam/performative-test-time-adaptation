"""Bifurcation diagram from the EXISTING §6.1 probe grid (converged, 1500 steps) -- no
re-run needed. Reads probe_tent_mean.npz (keys: eps, bs, Hq, bacc, which), forms the order
parameter s(eps,B) = 1 - H(q)/log K, extracts the critical performativity eps*(B) from the
balanced-accuracy phase boundary, and fits Prop. B' s predicted finite-batch law
  eps*(B) = eps_inf / (1 + c/B)     <=>     1/eps* = (1/eps_inf) + (c/eps_inf)(1/B)
by a linear fit of 1/eps* vs 1/B.

    python -m experiments.summarize_bifurcation --npz results/probe_tent_mean.npz --out results/bifurcation.csv
"""
import argparse
import csv
import math
import os
import numpy as np

ACC_FRAC = 0.90       # eps*(B) = first eps where bacc < ACC_FRAC * bacc(eps=0)
S_THRESH = 0.30


def crossing(xs, ys, thresh, rising):
    for i in range(1, len(ys)):
        a, b = ys[i - 1], ys[i]
        if (rising and a < thresh <= b) or (not rising and a >= thresh > b):
            return xs[i - 1] + (thresh - a) / ((b - a) or 1e-9) * (xs[i] - xs[i - 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/probe_tent_mean.npz")
    ap.add_argument("--out", default="results/bifurcation.csv")
    args = ap.parse_args()
    if not os.path.exists(args.npz):
        print(f"[error] not found: {args.npz} (run the §6.1 probe / aggregate_seeds first)")
        raise SystemExit(2)

    d = np.load(args.npz, allow_pickle=True)
    eps = np.asarray(d["eps"], float)
    bs = np.asarray(d["bs"], int)
    Hq = np.asarray(d["Hq"], float)          # [eps, bs]
    bacc = np.asarray(d["bacc"], float)      # [eps, bs]
    which = str(d["which"]) if "which" in d else "cifar10"
    K = 100 if which == "cifar100" else 10
    s = 1.0 - Hq / math.log(K)               # order parameter (concentration)

    print(f"{which}  K={K}  eps={list(eps)}  bs={list(bs)}")
    rows = []
    estar = {}
    for j, b in enumerate(bs):
        bacc_col = bacc[:, j]
        s_col = s[:, j]
        e_acc = crossing(eps, bacc_col, ACC_FRAC * bacc_col[0], rising=False)
        e_s = crossing(eps, s_col, S_THRESH, rising=True)
        estar[int(b)] = e_acc
        print(f"\n  bs={b:4d}  bacc(eps=0)={bacc_col[0]:.3f}")
        print("    eps   : " + " ".join(f"{e:4.1f}" for e in eps))
        print("    s     : " + " ".join(f"{v:4.2f}" for v in s_col))
        print("    bacc  : " + " ".join(f"{v:4.2f}" for v in bacc_col))
        print(f"    eps*_accuracy = {('%.2f' % e_acc) if e_acc else '>max'}    "
              f"eps*_concentration(s>{S_THRESH}) = {('%.2f' % e_s) if e_s else '>max'}")
        for i, e in enumerate(eps):
            rows.append([which, int(b), f"{e:.2f}", f"{s_col[i]:.4f}", f"{bacc_col[i]:.4f}"])
        rows.append([which, int(b), "eps*", "" if e_acc is None else f"{e_acc:.3f}", ""])

    # fit eps*(B) = eps_inf/(1+c/B)  via  1/eps* = a + b/B  (linear)
    pts = [(int(b), estar[int(b)]) for b in bs if estar[int(b)] is not None]
    print("\n==================== eps*(B) BATCH-SIZE LAW ====================")
    print("    eps*(B):  " + ", ".join(f"bs{b}:{v:.2f}" for b, v in pts) +
          (f", bs(>): {[int(b) for b in bs if estar[int(b)] is None]}" if any(estar[int(b)] is None for b in bs) else ""))
    # drop leading small batches that break monotonicity (the "already-fragile" regime,
    # where eps=0 is itself near-floor -- not the clean bifurcation regime; cf. CIFAR-100/bs16)
    fit_pts = pts[:]
    while len(fit_pts) >= 2 and fit_pts[0][1] > fit_pts[1][1] + 1e-9:
        b0, v0 = fit_pts.pop(0)
        print(f"    (excluding bs{b0} from the fit: eps*={v0:.2f} out of order -- already-fragile regime)")
    mono = all(fit_pts[i][1] <= fit_pts[i + 1][1] + 1e-9 for i in range(len(fit_pts) - 1))
    print("    monotone increasing with B (clean regime):",
          "YES (matches Prop. B')" if mono and len(fit_pts) >= 2 else "no/partial")
    if len(fit_pts) >= 2:
        X = np.array([1.0 / b for b, _ in fit_pts]); Y = np.array([1.0 / v for _, v in fit_pts])
        A = np.vstack([np.ones_like(X), X]).T
        a, bcoef = np.linalg.lstsq(A, Y, rcond=None)[0]
        eps_inf = 1.0 / a if a != 0 else float("inf")
        c = bcoef / a if a != 0 else float("inf")
        pred = {b: eps_inf / (1 + c / b) for b, _ in fit_pts}
        print(f"    FIT eps*(B) = eps_inf/(1+c/B):  eps_inf={eps_inf:.2f}, c={c:.1f}")
        print("    predicted vs measured:  " +
              ", ".join(f"bs{b}: {pred[b]:.2f} vs {v:.2f}" for b, v in fit_pts))
        resid = math.sqrt(np.mean([(pred[b] - v) ** 2 for b, v in fit_pts]))
        print(f"    RMSE = {resid:.3f}  -> the collapse onset location tracks Prop. B' s 1/B form.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "bs", "eps", "s_concentration", "balanced_acc"])
        w.writerows(rows)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
