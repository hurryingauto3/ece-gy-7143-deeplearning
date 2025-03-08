# Model Definition for ResNet Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary


class ResidualBlock(nn.Module):
    """
    Residual Block for ResNet architecture.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Stride for the first convolution. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First conv layer: 3x3 conv + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second conv layer: 3x3 conv + BatchNorm
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3,
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut for matching dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1,
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """Forward pass through the residual block."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet architecture with configurable number of blocks.
    
    Args:
        block (nn.Module): Block type to use (ResidualBlock)
        num_blocks (list): Number of blocks in each layer
        num_classes (int, optional): Number of output classes. Defaults to 10.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            3, 
            64, 
            kernel_size=3, 
            stride=1,
            padding=1, 
            bias=False
        )
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
        """
        Create a layer with multiple blocks.
        
        Args:
            block (nn.Module): Block type
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in this layer
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Sequential container of blocks
        """
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the ResNet."""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def train(model, optimizer, criterion, scheduler, train_loader, test_loader, device):
    """
    Train the model on the given data loaders.
    
    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        criterion (nn.Module): Loss function
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        device (torch.device): Device to train on
        
    Returns:
        tuple: (epoch_train_losses, epoch_test_losses)
    """
    epochs = 50
    final_accuracy = 0
    epoch_train_losses = []   # Average training loss per epoch
    epoch_test_losses = []    # Average test loss per epoch

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1

        # Step the scheduler
        scheduler.step()
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / num_train_batches
        epoch_train_losses.append(avg_train_loss)

        # Testing phase: compute both accuracy and loss
        model.eval()
        total_test_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        # Calculate test metrics
        avg_test_loss = total_test_loss / total
        epoch_test_losses.append(avg_test_loss)
        accuracy = 100.0 * correct / total

        print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        # Early stopping if accuracy is high enough
        if accuracy > 88:
            final_accuracy = accuracy
            break

    print(f"Final Accuracy: {final_accuracy:.2f}%")
    return epoch_train_losses, epoch_test_losses

