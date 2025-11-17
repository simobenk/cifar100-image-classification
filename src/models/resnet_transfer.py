import torch
import torch.nn as nn
from torchvision import models


def create_resnet18_model(
    num_classes: int = 100,
    mode: str = "feature_extract",  
):

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)



    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    if mode == "feature_extract":
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "finetune_last_block":
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "finetune_last_two_blocks":
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Mode inconnu: {mode}")

    return model


if __name__ == "__main__":
    model = create_resnet18_model(num_classes=100, mode="finetune_last_block")
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Nb params trainables:", sum(p.requires_grad for p in model.parameters()))
