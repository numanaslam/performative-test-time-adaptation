"""Per-step trajectories for the stability transition (reviewer Concern 7) and the
finite-batch activation-bias control (reviewer Concern 3).

One invocation runs ONE configuration of the performative loop and logs, at every
step, the soft marginal entropy H(q_hat), the PATTA gate value lambda_t that the
schedule WOULD apply, the induced-prior entropy, and the dominant predicted-class
share; at every eval window it probes balanced accuracy on a fixed class-uniform
set. The full arrays are saved to an .npz so the paper can plot HOW and WHEN
collapse develops, compare predicted- against oracle-label onset, and show the
distribution of lambda_t at K >> B.

Configurations (set by --adapter, --feedback, --adapt / --no-adapt):
  tent        , feedback none , adapt     : baseline, no feedback (eps ignored -> 0)
  tent        , feedback pred , adapt     : collapse trajectory (the paper's setting)
  patta_steep , feedback pred , adapt     : rescue trajectory
  tent        , feedback oracle, adapt    : oracle-label onset (true-label resampling)
  tent        , feedback pred , --no-adapt: BN-only control (test-time BN statistics,
                                            NO gradient updates) -- separates gradient-
                                            induced collapse from BN-statistics drift
  patta_steep , feedback none , adapt     : balanced-population lambda_t distribution
                                            (run at ImageNet scale, K >> B)

The lambda_t gate is recomputed here from the soft marginal with the same formula as
perftta.tta.Patta._lam (u = max(0,(Hmax-H(q_hat))/Hmax); lambda = lam0 * u**p), so
no core module is modified.

CIFAR rescue/onset/bn-only cell:
  python -m experiments.exp_trajectory --which cifar10 \
      --ckpt D:/data/checkpoints/resnet18_cifar10.pt --cdir D:/data/cifar10c/CIFAR-10-C \
      --corruption gaussian_noise --severity 5 --eps 1.0 --bs 16 --steps 1500 \
      --eval_every 25 --adapter tent --feedback pred --seed 0 \
      --out results/trajectory/traj_c10_gauss_tent_pred_s0.npz

ImageNet balanced-population lambda_t distribution (K=1000 >> B=16):
  python -m experiments.exp_trajectory --which imagenet --cdir D:/data/imagenet-c \
      --arch resnet50 --corruption contrast --severity 3 --eps 0.0 --bs 16 --steps 800 \
      --eval_every 100 --eval_bs 1000 --eval_chunk 128 --adapter patta_steep \
      --feedback none --p 0.5 --seed 0 \
      --out results/trajectory/traj_in_lambda_balanced_s0.npz
"""
import argparse
import math
import os
import numpy as np
import torch

from perftta.utils import get_device, marginal_entropy
from perftta.feedback import PerformativeFeedback
from perftta.metrics import accuracy, pred_hist, entropy_of_dist, max_class_share


def label_hist(y, K):
    """True-label marginal of the drawn batch (normalized)."""
    h = torch.bincount(y.detach().to("cpu").view(-1).long(), minlength=K).float()
    s = float(h.sum())
    return h / (s if s > 0 else 1.0)


def gate_lambda(Hq, Hmax, lam0, p):
    """PATTA's adaptive weight from the soft-marginal entropy -- mirrors Patta._lam."""
    u = max(0.0, (Hmax - Hq) / Hmax)
    return lam0 * (u ** p)


def build(which, name, args, device):
    """Fresh adapter (reset weights) and the shared pool for `which`."""
    if which in ("cifar10", "cifar100"):
        from perftta.datasets import CorruptedPool
        from experiments._common import fresh_adapter, num_classes_of
        K = num_classes_of(which)
        pool = build.pool or CorruptedPool(args.cdir, args.corruption, args.severity,
                                           num_classes=K, device=device)
        build.pool = pool
        ad = fresh_adapter(name, args.ckpt, K, device, lr=args.lr, optim=args.optim,
                           lam0=args.lam0, p=args.p)
        return ad, pool, K
    else:  # imagenet
        from perftta.scale_data import ImageFolderPool
        from perftta.models import build_imagenet_model
        from perftta.tta import build_adapter
        pool = build.pool or ImageFolderPool(args.cdir, args.corruption, args.severity,
                                             img_size=args.img_size, device=device,
                                             max_per_class=(args.max_per_class or None))
        build.pool = pool
        K = pool.K
        if K != 1000 and not args.allow_partial:
            raise SystemExit(f"[fatal] pool has {K} classes, not 1000; extract all wnids "
                             f"or pass --allow_partial (smoke only).")
        model = build_imagenet_model(args.arch).to(device)
        ad = build_adapter(name, model, num_classes=K, lr=args.lr, optim=args.optim,
                           lam0=args.lam0, p=args.p)
        return ad, pool, K
build.pool = None


def run(adapter, pool, K, args, device):
    g = torch.Generator().manual_seed(args.seed)
    eg = torch.Generator().manual_seed(args.seed + 10007)
    eps = 0.0 if args.feedback == "none" else args.eps
    fb = PerformativeFeedback(K, eps, None)
    prior = fb.base_prior.clone()
    uniform = torch.ones(K) / K
    Hmax = math.log(K)

    step_all, Hq_all, lam_all, priorH_all, share_all = [], [], [], [], []
    step_eval, bacc_eval = [], []

    for t in range(args.steps):
        x, y = pool.sample(prior, args.bs, generator=g)
        with torch.no_grad():
            logits = adapter.predict(x)
        probs = logits.detach().float().softmax(1)
        qbar = probs.mean(0)                                   # soft induced marginal
        Hq = float(marginal_entropy(qbar))
        lam = gate_lambda(Hq, Hmax, args.lam0, args.p)         # gate PATTA would apply

        step_all.append(t)
        Hq_all.append(Hq)
        lam_all.append(lam)
        priorH_all.append(entropy_of_dist(prior))
        share_all.append(max_class_share(pred_hist(logits.detach().cpu(), K)))

        if args.eval_every and ((t + 1) % args.eval_every == 0):
            xb, yb = pool.sample(uniform, args.eval_bs, generator=eg)
            with torch.no_grad():
                if args.eval_chunk and xb.shape[0] > args.eval_chunk:
                    lb = torch.cat([adapter.predict(xb[i:i + args.eval_chunk])
                                    for i in range(0, xb.shape[0], args.eval_chunk)], 0)
                else:
                    lb = adapter.predict(xb)
            step_eval.append(t + 1)
            bacc_eval.append(accuracy(lb, yb))

        if args.adapt:
            adapter.update(x)
        if args.feedback == "none":
            continue                                           # prior stays uniform
        fbh = pred_hist(logits.detach().cpu(), K) if args.feedback == "pred" else label_hist(y, K)
        prior = fb.next_prior(fbh)

    return dict(
        which=args.which, corruption=args.corruption, severity=args.severity,
        adapter=args.adapter, feedback=args.feedback, adapt=bool(args.adapt),
        eps=eps, bs=args.bs, seed=args.seed, logK=float(Hmax), lam0=args.lam0, p=args.p,
        step=np.array(step_all), Hq=np.array(Hq_all), lam=np.array(lam_all),
        prior_entropy=np.array(priorH_all), pred_share=np.array(share_all),
        step_eval=np.array(step_eval), bacc=np.array(bacc_eval),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100", "imagenet"])
    ap.add_argument("--ckpt", default="D:/data/checkpoints/resnet18_cifar10.pt")
    ap.add_argument("--cdir", default="D:/data/cifar10c/CIFAR-10-C")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--adapter", default="tent",
                    choices=["tent", "patta", "patta_steep", "source", "balance"])
    ap.add_argument("--feedback", default="pred", choices=["pred", "oracle", "none"])
    ap.add_argument("--adapt", dest="adapt", action="store_true", default=True)
    ap.add_argument("--no-adapt", dest="adapt", action="store_false")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--eval_chunk", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_per_class", type=int, default=0)
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--out", default="results/trajectory/traj.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device={device} which={args.which} corr={args.corruption} sev={args.severity} "
          f"adapter={args.adapter} feedback={args.feedback} adapt={args.adapt} "
          f"eps={args.eps} bs={args.bs} seed={args.seed}")

    ad, pool, K = build(args.which, args.adapter, args, device)
    log = run(ad, pool, K, args, device)

    nb = min(3, len(log["bacc"]))
    init_b = float(log["bacc"][0]) if len(log["bacc"]) else float("nan")
    fin_b = float(log["bacc"][-nb:].mean()) if len(log["bacc"]) else float("nan")
    print(f"  bal {init_b:.3f}->{fin_b:.3f}  "
          f"lam[min/med/max] {log['lam'].min():.2f}/{np.median(log['lam']):.2f}/{log['lam'].max():.2f}  "
          f"prior_H[end] {log['prior_entropy'][-1]:.2f}/{log['logK']:.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **log)
    print("saved", args.out)
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
