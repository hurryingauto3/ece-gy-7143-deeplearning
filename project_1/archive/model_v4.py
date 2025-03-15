import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.05):
        super(ResidualBlock, self).__init__()
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.bn1(self.conv1(x))
        out = F.relu(out, inplace=False)
        out = self.bn2(self.conv2(out))

        if self.training and torch.rand(1).item() < self.drop_prob:
            out = identity  # Skip residual connection randomly
        else:
            out = out + identity  # No in-place operation

        out = F.relu(out, inplace=False)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0.1):
        super(ResNet, self).__init__()
        self.in_channels = 32  # ✅ Lower Base Channels
        self.dropout = nn.Dropout(dropout_rate)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)

        # ✅ Reduced Channel Sizes & Blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 2 blocks
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 2 blocks
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 1 block
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)  # 1 block ✅ Reduced from 512 to 256

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)  # ✅ Reduced FC layer from 512 → 256

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out, inplace=False)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)  
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(ResidualBlock, [2, 2, 1, 1], dropout_rate=0.1).to(device)  # ✅ Reduced blocks
summary(model, (3, 32, 32))