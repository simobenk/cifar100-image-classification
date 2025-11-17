import torch
import torch.nn as nn
from torchvision import models


def create_resnet18_model(num_classes: int = 100, feature_extract: bool = True):

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = create_resnet18_model(num_classes=100, feature_extract=True)
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
