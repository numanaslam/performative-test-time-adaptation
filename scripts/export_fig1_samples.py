"""Export real dataset-image tiles for Figure 1 (methodology diagram).

Run ON THE GPU BOX (where D:/data lives). Needs only numpy + Pillow, no torch/torchvision.
It reads CIFAR-10-C .npy arrays and writes upscaled PNG tiles:
  - balanced_0..4.png : one sample from each of 5 different classes (a class-balanced stream)
  - collapsed_0..4.png : five samples of ONE dominant class (an adapter-induced collapsed stream)

    python scripts/export_fig1_samples.py --cdir D:/data/cifar10c/CIFAR-10-C --corruption gaussian_noise --severity 5

Then send the results/fig1/*.png files back; they drop straight into Figure 1 in main.tex.
"""
import argparse
import os
import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: pip install pillow")

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdir", default="D:/data/cifar10c/CIFAR-10-C",
                    help="folder with <corruption>.npy and labels.npy")
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5, choices=range(1, 6))
    ap.add_argument("--out", default="results/fig1")
    ap.add_argument("--upscale", type=int, default=128, help="output tile size in px")
    ap.add_argument("--dominant", type=int, default=3, help="class index for the collapsed row")
    args = ap.parse_args()

    xpath = os.path.join(args.cdir, args.corruption + ".npy")
    lpath = os.path.join(args.cdir, "labels.npy")
    if not (os.path.exists(xpath) and os.path.exists(lpath)):
        raise SystemExit(f"[fatal] missing {xpath} or {lpath}")

    X = np.load(xpath, mmap_mode="r")            # (50000, 32, 32, 3) uint8
    y = np.load(lpath)                           # (50000,)
    s = args.severity - 1
    sl = slice(s * 10000, (s + 1) * 10000)
    Xs, ys = np.asarray(X[sl]), np.asarray(y[sl])

    os.makedirs(args.out, exist_ok=True)

    def save(arr, name):
        im = Image.fromarray(arr.astype(np.uint8), "RGB").resize(
            (args.upscale, args.upscale), Image.NEAREST)
        p = os.path.join(args.out, name)
        im.save(p)
        print("saved", p)

    # balanced row: first sample of 5 distinct classes
    for i, cls in enumerate([0, 1, 2, 7, 8]):
        idx = np.where(ys == cls)[0][0]
        save(Xs[idx], f"balanced_{i}.png")

    # collapsed row: 5 samples of the dominant class (what the loop over-selects)
    dom_idx = np.where(ys == args.dominant)[0][:5]
    for i, idx in enumerate(dom_idx):
        save(Xs[idx], f"collapsed_{i}.png")

    print(f"\ndominant class = {CIFAR10_CLASSES[args.dominant]} ({args.corruption}, severity {args.severity})")
    print(f"done -> {args.out}  (send these PNGs back for Figure 1)")


if __name__ == "__main__":
    main()
