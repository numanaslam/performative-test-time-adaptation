"""Shared helpers. A fresh model is built per run because configure_tent()
mutates BN modules in place."""
from perftta.models import build_model
from perftta.tta import build_adapter


def fresh_model(ckpt, num_classes, device):
    return build_model(num_classes=num_classes, ckpt=ckpt, map_location="cpu").to(device)


def fresh_adapter(name, ckpt, num_classes, device, **kw):
    model = fresh_model(ckpt, num_classes, device)
    return build_adapter(name, model, num_classes=num_classes, **kw)


def num_classes_of(which):
    return 100 if which == "cifar100" else 10
