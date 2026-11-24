"""Compact CIFAR ResNet (BN-bearing, so TENT/PATTA can adapt BN affine params)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def build_model(num_classes=10, ckpt=None, map_location="cpu"):
    model = resnet18(num_classes)
    if ckpt is not None:
        state = torch.load(ckpt, map_location=map_location)
        state = state.get("model", state)
        model.load_state_dict(state)
    return model


def build_imagenet_model(arch="resnet50"):
    """Torchvision ImageNet-pretrained backbone for the scale experiment. BN-bearing
    (ResNet family) so the TENT/PATTA recipe adapts BN-affine params unchanged.
    Weights download to the torch hub cache on first use (on the GPU box)."""
    import torchvision.models as tvm
    arch = arch.lower()
    if arch == "resnet50":
        return tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    if arch == "resnet18":
        return tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    if arch == "resnet101":
        return tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1)
    raise ValueError(f"unknown imagenet arch: {arch!r} (use resnet18/resnet50/resnet101)")
