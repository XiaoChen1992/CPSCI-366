"""
PyTorch ResNet tutorial models
- ResNet-18 implemented in TWO ways:
  1) Plain/explicit version without block classes (for teaching the residual logic step-by-step).
  2) Canonical block-based implementation (BasicBlock / Bottleneck) that also powers ResNet-34/50/101.

All models accept (N, 3, 224, 224) by default and return (N, num_classes).

Test snippet at bottom to verify tensor shapes and parameter counts.
"""
from __future__ import annotations
from typing import Callable, List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


# ============================================================
# (A) Plain, explicit ResNet-18 WITHOUT block classes
#     — a didactic version showing each residual unit manually.
# ============================================================
class ResNet18Plain(nn.Module):
    """ResNet-18 implemented explicitly, layer by layer, no BasicBlock class.
    This is intentionally verbose for teaching.
    """
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        # Stem: 7x7 conv, BN, ReLU, 3x3 maxpool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Layer 1 (64 -> 64), two residual units, no downsampling
        # Unit 1
        self.l1_u1_conv1 = conv3x3(64, 64, stride=1)
        self.l1_u1_bn1   = nn.BatchNorm2d(64)
        self.l1_u1_conv2 = conv3x3(64, 64, stride=1)
        self.l1_u1_bn2   = nn.BatchNorm2d(64)
        # Unit 2
        self.l1_u2_conv1 = conv3x3(64, 64, stride=1)
        self.l1_u2_bn1   = nn.BatchNorm2d(64)
        self.l1_u2_conv2 = conv3x3(64, 64, stride=1)
        self.l1_u2_bn2   = nn.BatchNorm2d(64)

        # --- Layer 2 (64 -> 128), first unit downsamples (stride=2)
        # Unit 1 (with projection on the skip)
        self.l2_u1_conv1 = conv3x3(64, 128, stride=2)
        self.l2_u1_bn1   = nn.BatchNorm2d(128)
        self.l2_u1_conv2 = conv3x3(128, 128, stride=1)
        self.l2_u1_bn2   = nn.BatchNorm2d(128)
        self.l2_u1_proj  = conv1x1(64, 128, stride=2)
        self.l2_u1_proj_bn = nn.BatchNorm2d(128)
        # Unit 2
        self.l2_u2_conv1 = conv3x3(128, 128, stride=1)
        self.l2_u2_bn1   = nn.BatchNorm2d(128)
        self.l2_u2_conv2 = conv3x3(128, 128, stride=1)
        self.l2_u2_bn2   = nn.BatchNorm2d(128)

        # --- Layer 3 (128 -> 256)
        self.l3_u1_conv1 = conv3x3(128, 256, stride=2)
        self.l3_u1_bn1   = nn.BatchNorm2d(256)
        self.l3_u1_conv2 = conv3x3(256, 256, stride=1)
        self.l3_u1_bn2   = nn.BatchNorm2d(256)
        self.l3_u1_proj  = conv1x1(128, 256, stride=2)
        self.l3_u1_proj_bn = nn.BatchNorm2d(256)
        # Unit 2
        self.l3_u2_conv1 = conv3x3(256, 256, stride=1)
        self.l3_u2_bn1   = nn.BatchNorm2d(256)
        self.l3_u2_conv2 = conv3x3(256, 256, stride=1)
        self.l3_u2_bn2   = nn.BatchNorm2d(256)

        # --- Layer 4 (256 -> 512)
        self.l4_u1_conv1 = conv3x3(256, 512, stride=2)
        self.l4_u1_bn1   = nn.BatchNorm2d(512)
        self.l4_u1_conv2 = conv3x3(512, 512, stride=1)
        self.l4_u1_bn2   = nn.BatchNorm2d(512)
        self.l4_u1_proj  = conv1x1(256, 512, stride=2)
        self.l4_u1_proj_bn = nn.BatchNorm2d(512)
        # Unit 2
        self.l4_u2_conv1 = conv3x3(512, 512, stride=1)
        self.l4_u2_bn1   = nn.BatchNorm2d(512)
        self.l4_u2_conv2 = conv3x3(512, 512, stride=1)
        self.l4_u2_bn2   = nn.BatchNorm2d(512)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _residual(self, x, conv1, bn1, conv2, bn2, proj: nn.Module | None = None):
        identity = x
        out = self.relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        if proj is not None:
            # when downsampling/channel change
            identity = proj(identity)
        out += identity
        out = self.relu(out)
        return out

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1
        x = self._residual(x, self.l1_u1_conv1, self.l1_u1_bn1, self.l1_u1_conv2, self.l1_u1_bn2)
        x = self._residual(x, self.l1_u2_conv1, self.l1_u2_bn1, self.l1_u2_conv2, self.l1_u2_bn2)

        # Layer 2 (first unit uses projection)
        proj = nn.Sequential(self.l2_u1_proj, self.l2_u1_proj_bn)
        x = self._residual(x, self.l2_u1_conv1, self.l2_u1_bn1, self.l2_u1_conv2, self.l2_u1_bn2, proj)
        x = self._residual(x, self.l2_u2_conv1, self.l2_u2_bn1, self.l2_u2_conv2, self.l2_u2_bn2)

        # Layer 3
        proj = nn.Sequential(self.l3_u1_proj, self.l3_u1_proj_bn)
        x = self._residual(x, self.l3_u1_conv1, self.l3_u1_bn1, self.l3_u1_conv2, self.l3_u1_bn2, proj)
        x = self._residual(x, self.l3_u2_conv1, self.l3_u2_bn1, self.l3_u2_conv2, self.l3_u2_bn2)

        # Layer 4
        proj = nn.Sequential(self.l4_u1_proj, self.l4_u1_proj_bn)
        x = self._residual(x, self.l4_u1_conv1, self.l4_u1_bn1, self.l4_u1_conv2, self.l4_u1_bn2, proj)
        x = self._residual(x, self.l4_u2_conv1, self.l4_u2_bn1, self.l4_u2_conv2, self.l4_u2_bn2)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================
# (B) Canonical, reusable block-based ResNet
#     — powers 18/34 (BasicBlock) and 50/101 (Bottleneck).
# ============================================================
class BasicBlock(nn.Module):
    expansion = 1  # output_channels = planes * expansion

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        # 1x1 reduce
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1   = nn.BatchNorm2d(planes)
        # 3x3
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2   = nn.BatchNorm2d(planes)
        # 1x1 expand
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
                 num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        self.inplanes = 64

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ------------------------------------------------------------
# Factory functions for common ResNets
# ------------------------------------------------------------
def resnet18(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def resnet34(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def resnet50(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


def resnet101(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)


# ------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------
def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _demo():
    x = torch.randn(1, 3, 224, 224)

    print("Plain ResNet-18 (no blocks):")
    m_plain = ResNet18Plain(num_classes=10)
    y = m_plain(x)
    print("  output:", y.shape, "params:", _count_params(m_plain))

    print("\nBlock-based ResNet-18:")
    m18 = resnet18(num_classes=10)
    y = m18(x)
    print("  output:", y.shape, "params:", _count_params(m18))

    print("\nResNet-34 / 50 / 101:")
    for f in (resnet34, resnet50, resnet101):
        m = f(num_classes=10)
        y = m(x)
        print(f"  {f.__name__}: output {tuple(y.shape)}, params {_count_params(m)}")


if __name__ == "__main__":
    _demo()
