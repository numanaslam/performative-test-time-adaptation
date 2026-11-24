"""ImageNet-C scale pool: a class-indexed, LAZY pool over an ImageNet-C tree so the
performative loop can draw class-conditional batches at ImageNet scale (1000 classes,
224px real images) without loading the dataset into memory.

Tree layout (standard ImageNet-C extraction):
    <cdir>/<corruption>/<severity>/<wnid>/*.JPEG

Correctness note: a torchvision ImageNet-pretrained model's output index k corresponds
to the k-th wnid in SORTED order over all 1000 wnids. This pool uses the same sorted
order, so local class index == pretrained model class index -- ONLY when all 1000
wnids are present. With a subset the indices do not align and accuracy is meaningless;
exp_scale enforces K==1000 unless --allow_partial (smoke tests only).
"""
import os
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMG_EXT = (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG")


class ImageFolderPool:
    """Lazy class-indexed pool over <cdir>/<corruption>/<severity>/<wnid>/*.JPEG."""

    def __init__(self, cdir, corruption, severity, img_size=224, device="cpu",
                 max_per_class=None, transform=None):
        assert 1 <= severity <= 5
        base = os.path.join(cdir, corruption, str(severity))
        if not os.path.isdir(base):
            raise FileNotFoundError(f"ImageNet-C path not found: {base}")
        wnids = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        if not wnids:
            raise FileNotFoundError(f"no class folders under {base}")
        self.base = base
        self.wnids = wnids
        self.K = len(wnids)
        self.device = device
        if transform is None:                           # default = standard ImageNet eval transform
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize(256), T.CenterCrop(img_size),
                T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        self.tf = transform
        self.by_class = []
        for w in wnids:
            d = os.path.join(base, w)
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(_IMG_EXT)]
            files.sort()
            if max_per_class:
                files = files[:max_per_class]
            self.by_class.append(files)
        self._nonempty = [c for c in range(self.K) if self.by_class[c]]
        if not self._nonempty:
            raise FileNotFoundError(f"no images found under {base}")

    def _load(self, path):
        from PIL import Image
        with Image.open(path) as im:
            return self.tf(im.convert("RGB"))

    def sample(self, prior, batch_size, generator=None):
        prior = prior / prior.sum()
        classes = torch.multinomial(prior, batch_size, replacement=True, generator=generator)
        xs, ys = [], []
        for c in classes.tolist():
            pool = self.by_class[c]
            if not pool:                                    # class absent in this slice
                c = self._nonempty[int(torch.randint(len(self._nonempty), (1,),
                                                      generator=generator).item())]
                pool = self.by_class[c]
            j = int(torch.randint(len(pool), (1,), generator=generator).item())
            xs.append(self._load(pool[j]))
            ys.append(c)
        x = torch.stack(xs).to(self.device)
        y = torch.tensor(ys, dtype=torch.long, device=self.device)
        return x, y
