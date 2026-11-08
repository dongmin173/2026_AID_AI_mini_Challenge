"""Shared architecture and image processing for training and OFFLINE inference."""
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(arch="regnet32", pretrained=False):
    # Only the explicit training call may download ImageNet weights.
    if arch == "regnet32":
        weights = models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.regnet_y_3_2gf(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, 2)
    elif arch == "regnet16":
        weights = models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.regnet_y_1_6gf(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, 2)
    elif arch == "b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    count = sum(p.numel() for p in net.parameters())
    if count > 20_000_000:
        raise ValueError(f"Parameter limit exceeded: {count:,}")
    return net


def head_parameters(net):
    return net.fc.parameters() if hasattr(net, "fc") else net.classifier.parameters()


def image_transform(size=224, train=False):
    # Preserve almost the full photograph; avoid aggressive random crops.
    if train:
        ops = [transforms.Resize((size + 16, size + 16)),
               transforms.RandomCrop(size),
               transforms.RandomHorizontalFlip(),
               transforms.ColorJitter(0.15, 0.15, 0.15, 0.02)]
    else:
        ops = [transforms.Resize((size, size))]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def probability_from_logits(logits, temperature=1.0, flip_logits=None):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    p = torch.softmax(logits / temperature, dim=1)[:, 1]
    if flip_logits is not None:
        p = (p + torch.softmax(flip_logits / temperature, dim=1)[:, 1]) / 2
    return p
