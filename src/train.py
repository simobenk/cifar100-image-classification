import torch
import torch.nn as nn
import torch.optim as optim

from src.data.datasets import get_cifar100_dataloaders
from src.data.datasets import get_cifar100_dataloaders_resnet
from src.models.cnn_baseline import BaselineCNN
from src.models.cnn_enhanced import EnhancedCNN 
from src.models.resnet_transfer import create_resnet18_model

import argparse



def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS device (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=2,
                        help="Nombre d'epochs d'entraînement")

    parser.add_argument("--batch-size", type=int, default=64,
                        help="Taille de batch pour les DataLoaders")

    parser.add_argument(
        "--mode",
        type=str,
        default="finetune_last_block",
        choices=["feature_extract", "finetune_last_block", "finetune_last_two_blocks"],
        help="Mode de fine-tuning du ResNet18",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # train_loader, test_loader = get_cifar100_dataloaders(batch_size=64, num_workers=2)
    train_loader, test_loader = get_cifar100_dataloaders_resnet(batch_size=args.batch_size, num_workers=2)
    # model = BaselineCNN(num_classes=100).to(device)
    # model = EnhancedCNN(num_classes=100).to(device) 
    model = create_resnet18_model(
        num_classes=100, 
        mode=args.mode,
        ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  

        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    print(f"Params backbone: {len(backbone_params)}, head: {len(head_params)}")

    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": 1e-4},  
            {"params": head_params, "lr": 1e-3},      
        ]
    )


    num_epochs = args.epochs

    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        avg_test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"- Train loss: {avg_train_loss:.4f} "
            f"- Test loss: {avg_test_loss:.4f} "
            f"- Test acc: {test_acc*100:.2f}%"
        )

if __name__ == "__main__":
    main()
