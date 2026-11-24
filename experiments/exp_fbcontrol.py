"""Feedback-generator controls (reviewer Issue 1): is the balanced-accuracy collapse a
GENERATOR artifact (the hard-label, finite-batch prior can be supported on <= B classes,
so the induced prior concentrates by construction) or genuine ADAPTATION instability?

We run the performative loop at the collapse cell (eps=1) in four modes and read, for each,
the induced-prior entropy (does the GENERATOR concentrate the prior?) and the balanced
accuracy on the fixed class-uniform set (does the MODEL actually deteriorate?):

  adapt_pred    TENT adapts; feedback = model's HARD predictions   (the paper's setting)
  frozen_pred   model FROZEN (source); feedback = frozen preds      (generator still runs)
  adapt_oracle  TENT adapts; feedback = TRUE-label marginal          (no prediction->prior coupling)
  frozen_oracle model FROZEN; feedback = TRUE-label marginal         (pure sanity)

Reading:
  * frozen_pred: if prior_entropy << log K (generator concentrated the prior) but balanced
    accuracy stays ~ its initial value, the generator's concentration ALONE does not cause
    the balanced-accuracy collapse -- it is adaptation-driven.
  * adapt_oracle: if adapting on a true-label-driven stream does NOT collapse while adapt_pred
    does, the collapse specifically needs PREDICTION-dependent feedback (the performative
    mechanism), not mere resampling.

CIFAR:
  python -m experiments.exp_fbcontrol --which cifar10 --ckpt D:/data/checkpoints/resnet18_cifar10.pt \
      --cdir D:/data/cifar10c/CIFAR-10-C --corruption gaussian_noise --severity 5 \
      --eps 1.0 --bs 16 --steps 1500 --seed 0 --out results/fbcontrol/fbc_c10_gaussian_s0.npz

ImageNet-C (flagship -- the contrast/sev3 clean-collapse cell):
  python -m experiments.exp_fbcontrol --which imagenet --cdir D:/data/imagenet-c --arch resnet50 \
      --corruption contrast --severity 3 --eps 1.0 --bs 16 --steps 1000 --seed 0 \
      --eval_every 200 --eval_bs 1000 --eval_chunk 128 --out results/fbcontrol/fbc_in_contrast_sev3_s0.npz
"""
import argparse
import os
import numpy as np
import torch

from perftta.utils import get_device
from perftta.feedback import PerformativeFeedback
from perftta.metrics import accuracy, pred_hist, entropy_of_dist

# mode -> (adapter name, feedback source, whether the model adapts)
MODES = {
    "adapt_pred":   ("tent",   "pred",   True),
    "frozen_pred":  ("source", "pred",   False),
    "adapt_oracle": ("tent",   "oracle", True),
    "frozen_oracle":("source", "oracle", False),
}


def label_hist(y, K):
    """True-label marginal of the drawn batch (normalized), matching pred_hist's shape."""
    h = torch.bincount(y.detach().to("cpu").view(-1).long(), minlength=K).float()
    s = float(h.sum())
    return h / (s if s > 0 else 1.0)


def build(which, name, args, device):
    """Return a fresh adapter (reset weights) and the shared pool for `which`."""
    if which in ("cifar10", "cifar100"):
        from perftta.datasets import CorruptedPool
        from experiments._common import fresh_adapter, num_classes_of
        K = num_classes_of(which)
        pool = build.pool or CorruptedPool(args.cdir, args.corruption, args.severity,
                                           num_classes=K, device=device)
        build.pool = pool
        ad = fresh_adapter(name, args.ckpt, K, device, lr=args.lr, optim=args.optim)
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
            raise SystemExit(f"[fatal] pool has {K} classes, not 1000; extract all wnids or pass --allow_partial (smoke only).")
        model = build_imagenet_model(args.arch).to(device)
        ad = build_adapter(name, model, num_classes=K, lr=args.lr, optim=args.optim)
        return ad, pool, K
build.pool = None


def run_control(adapter, pool, K, eps, steps, bs, seed, feedback, adapt,
                eval_every, eval_bs, eval_chunk):
    g = torch.Generator().manual_seed(seed)
    eg = torch.Generator().manual_seed(seed + 10007)
    fb = PerformativeFeedback(K, eps, None)
    prior = fb.base_prior.clone()
    uniform = torch.ones(K) / K
    baccs, pent, hqs = [], [], []
    for t in range(steps):
        x, y = pool.sample(prior, bs, generator=g)
        with torch.no_grad():
            logits = adapter.predict(x)
        hp = pred_hist(logits.detach().cpu(), K)          # hard predicted-class marginal
        pent.append(entropy_of_dist(prior))                # entropy of the INDUCED prior
        hqs.append(entropy_of_dist(hp))
        if eval_every and ((t + 1) % eval_every == 0):     # balanced probe on a class-uniform set
            xb, yb = pool.sample(uniform, eval_bs, generator=eg)
            with torch.no_grad():
                if eval_chunk and xb.shape[0] > eval_chunk:
                    lb = torch.cat([adapter.predict(xb[i:i + eval_chunk])
                                    for i in range(0, xb.shape[0], eval_chunk)], 0)
                else:
                    lb = adapter.predict(xb)
            baccs.append(accuracy(lb, yb))
        if adapt:
            adapter.update(x)
        fbh = hp if feedback == "pred" else label_hist(y, K)
        prior = fb.next_prior(fbh)

    nb, ne = min(3, len(baccs)), min(10, len(pent))
    return dict(
        init_bacc=(baccs[0] if baccs else float("nan")),
        final_bacc=(sum(baccs[-nb:]) / nb if baccs else float("nan")),
        final_prior_entropy=(sum(pent[-ne:]) / ne),
        final_Hq=(sum(hqs[-ne:]) / ne),
        logK=float(np.log(K)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100", "imagenet"])
    ap.add_argument("--ckpt", default="D:/data/checkpoints/resnet18_cifar10.pt")  # cifar
    ap.add_argument("--cdir", default="D:/data/cifar10c/CIFAR-10-C")
    ap.add_argument("--arch", default="resnet50")                                 # imagenet
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--modes", default="adapt_pred,frozen_pred,adapt_oracle,frozen_oracle")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--eval_chunk", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_per_class", type=int, default=0)
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--out", default="results/fbcontrol/fbcontrol.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"device={device} which={args.which} corr={args.corruption} sev={args.severity} "
          f"eps={args.eps} bs={args.bs} seed={args.seed}")

    rows = []
    for mode in modes:
        name, feedback, adapt = MODES[mode]
        ad, pool, K = build(args.which, name, args, device)
        log = run_control(ad, pool, K, args.eps, args.steps, args.bs, args.seed,
                          feedback, adapt, args.eval_every, args.eval_bs, args.eval_chunk)
        rows.append((mode, log))
        print(f"  [{mode:13s}] bal {log['init_bacc']:.3f}->{log['final_bacc']:.3f}  "
              f"prior_H {log['final_prior_entropy']:.3f}/{log['logK']:.3f}  H(q) {log['final_Hq']:.3f}")
        del ad
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, which=args.which, corruption=args.corruption, severity=args.severity,
             seed=args.seed, eps=args.eps, bs=args.bs,
             modes=np.array([r[0] for r in rows]),
             init_bacc=np.array([r[1]["init_bacc"] for r in rows]),
             final_bacc=np.array([r[1]["final_bacc"] for r in rows]),
             final_prior_entropy=np.array([r[1]["final_prior_entropy"] for r in rows]),
             final_Hq=np.array([r[1]["final_Hq"] for r in rows]),
             logK=rows[0][1]["logK"])

    d = {m: lg for m, lg in rows}
    print("\n--- reading ---")
    if "frozen_pred" in d:
        fp = d["frozen_pred"]
        conc = "YES" if fp["final_prior_entropy"] < 0.8 * fp["logK"] else "no"
        flat = "YES" if abs(fp["final_bacc"] - fp["init_bacc"]) < 0.05 else "no"
        print(f"frozen_pred:  generator concentrated the prior? {conc} "
              f"(H {fp['final_prior_entropy']:.2f} vs log K {fp['logK']:.2f});  "
              f"balanced accuracy stayed flat? {flat} "
              f"({fp['init_bacc']:.3f}->{fp['final_bacc']:.3f})  "
              f"=> if YES/YES, the collapse is NOT a generator artifact.")
    if "adapt_pred" in d and "adapt_oracle" in d:
        ap_, ao = d["adapt_pred"], d["adapt_oracle"]
        print(f"adapt_pred vs adapt_oracle: {ap_['final_bacc']:.3f} vs {ao['final_bacc']:.3f} "
              f"=> if oracle >> pred, the collapse needs PREDICTION-dependent feedback.")
    print("saved", args.out)


if __name__ == "__main__":
    main()
