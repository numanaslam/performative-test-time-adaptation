"""Export HD ImageNet-C tiles for Figure 1 (methodology diagram).

Run ON THE GPU BOX (where the ImageNet-C tree lives). Needs only Pillow -- no torch.
ImageNet-C is a folder tree:  <root>/<corruption>/<severity>/<wnid>/*.JPEG  (images are ~224 px, HD).

It writes:
  balanced_0..4.png : one image from each of 5 different classes (a class-balanced stream)
  collapsed_0..4.png : five images of ONE class (an adapter-induced collapsed stream)

Usage (adjust --root to wherever ImageNet-C sits on the box):
    python scripts/export_fig1_imagenetc.py --root D:/data/imagenet_c --corruption gaussian_noise --severity 3

If --root is wrong the script prints the candidate paths it tried and what it found.
Then send the results/fig1_hd/*.png files back; they replace the CIFAR tiles in Figure 1.
"""
import argparse
import glob
import os

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: pip install pillow")

EXTS = ("*.JPEG", "*.jpeg", "*.jpg", "*.JPG", "*.png", "*.PNG")
ROOT_CANDIDATES = [
    "D:/data/imagenet_c", "D:/data/imagenetc", "D:/data/ImageNet-C",
    "D:/data/imagenet-c", "D:/data/imagenet_c/ImageNet-C",
]


def imgs_in(d):
    out = []
    for e in EXTS:
        out += glob.glob(os.path.join(d, e))
    return sorted(out)


def resolve_base(root, corruption, severity):
    for base in [os.path.join(root, corruption, str(severity)),
                 os.path.join(root, "ImageNet-C", corruption, str(severity))]:
        if os.path.isdir(base):
            return base
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="ImageNet-C root (tries common paths if omitted)")
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=3, choices=range(1, 6))
    ap.add_argument("--out", default="results/fig1_hd")
    ap.add_argument("--size", type=int, default=256, help="output tile size in px")
    args = ap.parse_args()

    roots = [args.root] if args.root else ROOT_CANDIDATES
    base = None
    for r in roots:
        if r and (base := resolve_base(r, args.corruption, args.severity)):
            break
    if base is None:
        print("[fatal] could not find ImageNet-C. Tried:")
        for r in roots:
            print("   ", os.path.join(r or "", args.corruption, str(args.severity)))
        print("Pass the correct --root (the folder that CONTAINS the corruption subfolders).")
        raise SystemExit(1)
    print("using:", base)

    class_dirs = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
    if len(class_dirs) < 5:
        raise SystemExit(f"[fatal] expected class subfolders under {base}, found {len(class_dirs)}")

    os.makedirs(args.out, exist_ok=True)

    def save(path_in, name):
        im = Image.open(path_in).convert("RGB")
        s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2,
                      (im.width - s) // 2 + s, (im.height - s) // 2 + s))
        im = im.resize((args.size, args.size), Image.LANCZOS)
        p = os.path.join(args.out, name)
        im.save(p)
        print("saved", p)

    # balanced: first image of 5 distinct classes
    picked = 0
    for d in class_dirs:
        ims = imgs_in(d)
        if ims:
            save(ims[0], f"balanced_{picked}.png")
            picked += 1
        if picked == 5:
            break

    # collapsed: 5 images from the first class that has >=5
    for d in class_dirs:
        ims = imgs_in(d)
        if len(ims) >= 5:
            for i in range(5):
                save(ims[i], f"collapsed_{i}.png")
            print(f"\ncollapsed class folder = {os.path.basename(d)}")
            break

    print(f"done -> {args.out}  ({args.corruption}, severity {args.severity})  send these PNGs back")


if __name__ == "__main__":
    main()
