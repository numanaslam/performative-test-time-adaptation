"""Clean CIFAR loaders and a class-indexed CIFAR-C pool for prior-conditioned
sampling (the mechanism the performative map samples through)."""
import os
import numpy as np
import torch
from .utils import CIFAR_MEAN, CIFAR_STD

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


def _normalize(x_uint8):
    """(N,32,32,3) uint8 -> normalized (N,3,32,32) float tensor."""
    x = torch.from_numpy(np.ascontiguousarray(x_uint8)).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class CorruptedPool:
    """One corruption + severity slice of CIFAR-{10,100}-C, indexed by class so
    we can draw a batch from any class prior pi: y ~ pi, x ~ P(x|y)."""

    def __init__(self, cdir, corruption, severity, num_classes=10, device="cpu"):
        assert 1 <= severity <= 5
        X = np.load(os.path.join(cdir, corruption + ".npy"))        # (50000,32,32,3)
        y = np.load(os.path.join(cdir, "labels.npy")).astype(np.int64)  # (50000,)
        lo, hi = (severity - 1) * 10000, severity * 10000
        self.X = _normalize(X[lo:hi])
        self.y = torch.from_numpy(y[lo:hi]).long()
        self.K = num_classes
        self.device = device
        self.by_class = [(self.y == c).nonzero(as_tuple=True)[0] for c in range(num_classes)]

    def sample(self, prior, batch_size, generator=None):
        prior = prior / prior.sum()
        classes = torch.multinomial(prior, batch_size, replacement=True, generator=generator)
        return self.draw(classes.tolist(), generator=generator)

    def draw(self, labels, generator=None):
        """Draw one image per requested class label (an explicit label sequence), e.g.
        from a temporally-correlated stream. Returns (X, Y) with Y == labels."""
        idx = torch.empty(len(labels), dtype=torch.long)
        for i, c in enumerate(labels):
            pool = self.by_class[c]
            if len(pool) == 0:  # class absent in this slice (possible for CIFAR-100 subsets)
                idx[i] = torch.randint(len(self.y), (1,), generator=generator).item()
            else:
                j = torch.randint(len(pool), (1,), generator=generator).item()
                idx[i] = pool[j]
        return self.X[idx].to(self.device), self.y[idx].to(self.device)


def load_clean(root, which="cifar10", train=True):
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    cls = torchvision.datasets.CIFAR10 if which == "cifar10" else torchvision.datasets.CIFAR100
    return cls(root, train=train, download=True, transform=tf)
