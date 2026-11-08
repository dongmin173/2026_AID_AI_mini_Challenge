"""Model definitions and transforms. Offline inference always uses pretrained=False."""
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, efficientnet_b2,
    EfficientNet_B0_Weights, EfficientNet_B2_Weights,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_model(arch='b0', pretrained=False):
    if arch == 'b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    elif arch == 'b2':
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError('Only b0 and b2 supported: ' + arch)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def get_transform(size=224, train=False):
    ops = [transforms.Resize((size, size))]
    if train:
        # Same mild augmentations as original Colab notebook; no external data.
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.15, hue=0.02),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
