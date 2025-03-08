import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary

# Command-line arguments
import argparse

parser = argparse.ArgumentParser(description="ResNet Training on CIFAR-10")
parser.add_argument(
    "--batch_size", "-b", type=int, default=128, help="training batch size"
)
parser.add_argument(
    "--test_batch_size", "-tb", type=int, default=100, help="testing batch size"
)
parser.add_argument(
    "--epochs", "-e", type=int, default=30, help="number of epochs to train"
)
parser.add_argument("--lr", "-l", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--num_workers", "-nw", type=int, default=2, help="number of dataloader workers")
parser.add_argument(
    "--weight_decay", "-wd", type=float, default=5e-4, help="weight decay"
)
parser.add_argument(
    "--optimizer",
    "-o",
    type=str,
    default="Adam",
    choices=["SGD", "Adam", "RMSProp", "AdaFactor"],
    help="optimizer",
)
parser.add_argument("--device", "-d", default="cuda", type=str, help="training device")
args = parser.parse_args()

# Device
if args.device == "mps" and torch.backends.mps.is_available():
    device = torch.device("mps")
elif args.device == "cuda" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Model Definition
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First conv layer: 3x3 conv + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second conv layer: 3x3 conv + BatchNorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Shortcut for matching dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Four layers with varying output channels and strides
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Global average pooling and a fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Data Loaders
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainloader = DataLoader(
    torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    ),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)
testloader = DataLoader(
    torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    ),
    batch_size=args.test_batch_size,
    num_workers=args.num_workers,
    shuffle=False,
)

# Model initialization
model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(args.device)

# Optimizer and Scheduler
optimizer_dict = {
    "SGD": optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay
    ),
    "Adam": optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay),
    "RMSProp": optim.RMSprop(
        model.parameters(), lr=0.001, weight_decay=args.weight_decay
    ),
    "Adafactor": optim.Adafactor(
        model.parameters(), lr=0.001, weight_decay=args.weight_decay
    ),
}

optimizer = optimizer_dict[args.optimizer]
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

# Model Summary

if args.device == "mps":
    summary(model.to("cpu"), (3, 32, 32))
    model.to("mps")
else:
    summary(model, (3, 32, 32))

# Training Loop
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(trainloader, desc=f"Epoch {epoch}"):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%, Loss: {total_loss:.2f}")

