from __future__ import annotations

from math import prod

import torch
from torch import nn


class Classifier(nn.Module):
    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearClassifier(Classifier):
    def __init__(self, input_shape: tuple[int, ...], classes: int) -> None:
        super().__init__()
        self.output = nn.Linear(prod(input_shape), classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


class MLPClassifier(Classifier):
    def __init__(self, input_shape: tuple[int, ...], hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(prod(input_shape), hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.hidden(torch.flatten(x, start_dim=1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


class SmallCNNClassifier(Classifier):
    def __init__(self, input_shape: tuple[int, ...], hidden_dim: int, classes: int) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("small_cnn expects input shape (channels, height, width)")
        channels, height, width = input_shape
        self.convolution = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flattened = 64 * (height // 4) * (width // 4)
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(flattened, hidden_dim), nn.ReLU())
        self.output = nn.Linear(hidden_dim, classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolution(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


class SmallCNNBatchNormClassifier(Classifier):
    """Small CNN with train/eval-sensitive normalization for method-validity pilots."""

    def __init__(self, input_shape: tuple[int, ...], hidden_dim: int, classes: int) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("small_cnn_bn expects input shape (channels, height, width)")
        channels, height, width = input_shape
        self.convolution = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flattened = 64 * (height // 4) * (width // 4)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_dim, classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolution(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(x) + self.shortcut(x))


class ResNet18Classifier(Classifier):
    """CIFAR-sized ResNet-18 with a stable pooled feature representation."""

    def __init__(self, input_shape: tuple[int, ...], hidden_dim: int, classes: int) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("resnet18 expects input shape (channels, height, width)")
        channels, _, _ = input_shape
        width = hidden_dim
        if width < 8:
            raise ValueError("resnet18 hidden_dim must be at least 8")
        self.stem = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._layer(width, width, stride=1)
        self.layer2 = self._layer(width, width * 2, stride=2)
        self.layer3 = self._layer(width * 2, width * 4, stride=2)
        self.layer4 = self._layer(width * 4, width * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(width * 8, classes)

    @staticmethod
    def _layer(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.pool(x), start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


def build_model(
    architecture: str, input_shape: int | tuple[int, ...], hidden_dim: int, classes: int
) -> Classifier:
    shape = (input_shape,) if isinstance(input_shape, int) else input_shape
    if architecture == "linear":
        return LinearClassifier(shape, classes)
    if architecture == "mlp":
        return MLPClassifier(shape, hidden_dim, classes)
    if architecture == "small_cnn":
        return SmallCNNClassifier(shape, hidden_dim, classes)
    if architecture == "small_cnn_bn":
        return SmallCNNBatchNormClassifier(shape, hidden_dim, classes)
    if architecture == "resnet18":
        return ResNet18Classifier(shape, hidden_dim, classes)
    raise ValueError(f"unsupported architecture: {architecture}")
