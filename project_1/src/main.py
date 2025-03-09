"""
Main script for training the ResNet model on CIFAR-10 dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from project_1.src.model_training import ResNet, ResidualBlock, train
from data import CIFAR10DataModule, display_all_data, generate_submission


if __name__ == "__main__":
    # Set up device for training
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Initialize ResNet model - ResNet18 architecture
    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
    summary(model, (3, 32, 32))

    # Setup optimizer, scheduler and loss function
    weight_decay = float("5e-4")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()

    # Setup data loaders
    data_module = CIFAR10DataModule(batch_size=128)
    train_loader = data_module.get_train_loader()
    test_loader = data_module.get_test_loader()