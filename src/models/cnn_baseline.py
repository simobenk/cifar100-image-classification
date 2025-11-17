import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,  
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,  
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Après 2 pools :
        # Input : 3x32x32
        # conv1 -> 16x32x32
        # pool -> 16x16x16
        # conv2 -> 32x16x16
        # pool -> 32x8x8
        # donc taille du flatten = 32 * 8 * 8 = 2048

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1) 

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  

        return x


if __name__ == "__main__":
    model = BaselineCNN(num_classes=100)

    dummy_input = torch.randn(8, 3, 32, 32)  
    outputs = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", outputs.shape)
