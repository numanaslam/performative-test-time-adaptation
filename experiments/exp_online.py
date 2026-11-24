"""E3 - Online accuracy & collapse across corruptions and adapters.

    python -m experiments.exp_online --ckpt D:/data/checkpoints/resnet18_cifar10.pt \
        --cdir D:/data/cifar10c/CIFAR-10-C --corruptions gaussian_noise,fog,contrast \
        --adapters source,tent,eata_lite,patta --eps 0.6
"""
import argparse
import csv
import os

from perftta.datasets import CorruptedPool, CORRUPTIONS
from perftta.loop import run_performative_loop
from perftta.utils import get_device
from experiments._common import fresh_adapter, num_classes_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/data/checkpoints/resnet18_cifar10.pt")
    ap.add_argument("--cdir", default="D:/data/cifar10c/CIFAR-10-C")
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--corruptions", default="gaussian_noise,fog,contrast",
                    help="comma list, or 'all'")
    ap.add_argument("--adapters", default="source,tent,eata_lite,patta")
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/online.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    K = num_classes_of(args.which)
    corrs = CORRUPTIONS if args.corruptions == "all" else args.corruptions.split(",")
    adapters = args.adapters.split(",")

    rows = []
    for corr in corrs:
        pool = CorruptedPool(args.cdir, corr, args.severity, num_classes=K, device=device)
        for name in adapters:
            ad = fresh_adapter(name, args.ckpt, K, device, lr=args.lr, optim="sgd")
            log = run_performative_loop(ad, pool, K, args.eps, args.steps, args.bs, seed=args.seed)
            rows.append((corr, name, log["mean_acc"], log["final_Hq"], log["ttc"]))
            print(f"{corr:16s} {name:10s} acc={log['mean_acc']:.3f} "
                  f"final_Hq={log['final_Hq']:.3f} ttc={log['ttc']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "adapter", "mean_acc", "final_Hq", "ttc"])
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
