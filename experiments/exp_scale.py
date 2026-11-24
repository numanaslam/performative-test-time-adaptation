"""SCALE experiment: does the performative collapse / hidden-metric / PATTA-recovery
story hold at ImageNet scale (1000 classes, 224px, real images, ResNet-50)?

Runs the performative loop for each adapter at one (eps, batch) cell on ImageNet-C and
records final balanced accuracy, stream accuracy, H(q). 1000 classes make the hidden-
metric effect extreme (chance = 0.001).

    python -m experiments.exp_scale --cdir D:/data/imagenet-c --arch resnet50 \
        --corruption gaussian_noise --eps 1.0 --bs 16 --steps 1000 --seed 0 \
        --out results/scale/scale_in_gaussian_noise_bs16_s0.npz
"""
import argparse
import os
import numpy as np

from perftta.scale_data import ImageFolderPool
from perftta.models import build_imagenet_model
from perftta.loop import run_performative_loop
from perftta.utils import get_device
from perftta.tta import build_adapter


def fresh_adapter_scale(name, arch, K, device, **kw):
    # a fresh pretrained model per adapter: configure_tent() mutates BN modules in place
    model = build_imagenet_model(arch).to(device)
    return build_adapter(name, model, num_classes=K, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdir", default="D:/data/imagenet-c")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--adapters", default="source,tent,balance,patta,patta_sev")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_bs", type=int, default=1000)
    ap.add_argument("--eval_chunk", type=int, default=128)       # chunk the eval forward (ResNet-50/224px OOMs on a big single forward)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_per_class", type=int, default=0)   # 0 = all (ImageNet-C val: 50/class)
    ap.add_argument("--allow_partial", action="store_true")   # SMOKE ONLY: K<1000 -> wrong labels
    ap.add_argument("--out", default="results/scale/scale.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    pool = ImageFolderPool(args.cdir, args.corruption, args.severity,
                           img_size=args.img_size, device=device,
                           max_per_class=(args.max_per_class or None))
    K = pool.K
    if K != 1000 and not args.allow_partial:
        raise SystemExit(
            f"[fatal] pool has {K} classes, not 1000 -> local labels will NOT align with the "
            f"pretrained {args.arch} output index, so accuracy is meaningless. Extract all 1000 "
            f"wnids for this corruption/severity, or pass --allow_partial for a SMOKE test only.")
    names = [a.strip() for a in args.adapters.split(",") if a.strip()]
    print(f"device={device} arch={args.arch} K={K} corr={args.corruption} sev={args.severity} "
          f"eps={args.eps} bs={args.bs} seed={args.seed}")

    import torch
    bacc, sacc, hq, ibacc = [], [], [], []
    for name in names:
        ad = fresh_adapter_scale(name, args.arch, K, device, lr=args.lr, optim=args.optim)
        log = run_performative_loop(ad, pool, K, args.eps, args.steps, args.bs, seed=args.seed,
                                    eval_every=args.eval_every, eval_bs=args.eval_bs,
                                    eval_chunk=args.eval_chunk)
        bacc.append(log["final_bacc"]); sacc.append(log["final_acc"])
        hq.append(log["final_Hq"]); ibacc.append(log["init_bacc"])
        print(f"  [{name:10s}] bal={log['init_bacc']:.3f}->{log['final_bacc']:.3f}  "
              f"stream={log['final_acc']:.3f}  H(q)={log['final_Hq']:.3f}")
        del ad                                                   # free the model before the next adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, which="imagenet", arch=args.arch, corruption=args.corruption,
             severity=args.severity, seed=args.seed, eps=args.eps, bs=args.bs,
             adapters=np.array(names), final_bacc=np.array(bacc), final_acc=np.array(sacc),
             final_Hq=np.array(hq), init_bacc=np.array(ibacc))
    print("saved", args.out)


if __name__ == "__main__":
    main()
