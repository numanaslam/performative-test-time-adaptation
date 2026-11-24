"""Train a source model on CLEAN CIFAR and save a checkpoint.

Run on the 5070 box (needs torch+torchvision):
    python -m experiments.train_source --which cifar10 --epochs 30 \
        --out data/checkpoints/resnet18_cifar10.pt
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from perftta.models import resnet18
from perftta.datasets import load_clean
from perftta.utils import get_device, set_seed
from experiments._common import num_classes_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--root", default="D:/data/cifar")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out", default="D:/data/checkpoints/resnet18_cifar10.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    K = num_classes_of(args.which)
    print(f"device={device} classes={K}")

    train = load_clean(args.root, args.which, train=True)
    test = load_clean(args.root, args.which, train=False)
    tl = DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(test, batch_size=256, shuffle=False, num_workers=4)

    model = resnet18(K).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.numel()
        print(f"epoch {ep+1}/{args.epochs}  clean-test acc {correct/total:.4f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "which": args.which}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
