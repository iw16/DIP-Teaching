import torch
import torch.nn as nn
import torch.nn.functional as F

from FCN_network import FullyConvNetwork as Generator

class Discriminator(nn.Module):

    def __init__(self) -> None:
        C: int = 3
        K: int = 4
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=(C * 2),  out_channels=8,  kernel_size=K, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8,  out_channels=16, kernel_size=K, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=K, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=K, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=K, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
        )
        self.convf = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=K, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy: torch.Tensor = torch.cat([x, y], dim=1)
        xy = self.conv1(xy)
        xy = self.conv2(xy)
        xy = self.conv3(xy)
        xy = self.conv4(xy)
        xy = self.conv5(xy)
        xy = self.convf(xy)
        xy = F.sigmoid(xy)
        return xy