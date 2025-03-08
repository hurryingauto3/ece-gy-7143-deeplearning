#!/usr/bin/env python
import argparse
import datetime
import os
import zipfile
import torch
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ResNet, ResidualBlock

def download_and_prepare_data():
    dataset_dir = "./data"
    competition_name = "deep-learning-spring-2025-project-1"
    competition_path = os.path.join(dataset_dir, competition_name)
    zip_path = os.path.join(competition_path, f"{competition_name}.zip")
    
    # Expected structure after extraction:
    # competition_path/
    #   cifar-10-python/
    #     cifar-10-batches-py/
    #       data_batch_1, ..., test_batch, batches.meta, ...
    cifar_python_dir = os.path.join(competition_path, "cifar-10-python")
    cifar_dir = os.path.join(cifar_python_dir, "cifar-10-batches-py")

    # Download and extract if not already available
    if not os.path.exists(cifar_dir):
        os.makedirs(competition_path, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(competition_name, path=competition_path)
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(competition_path)
    
    if not os.path.exists(cifar_dir):
        raise FileNotFoundError(f"Dataset not found in '{cifar_dir}'")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Set root to the folder containing "cifar-10-batches-py"
    test_dataset = datasets.CIFAR10(
        root=cifar_python_dir,
        train=False,
        transform=test_transform,
        download=False
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

def generate_kaggle_predictions(model_path):
    test_loader = download_and_prepare_data()

    # Initialize the model structure before loading weights
    model = ResNet(ResidualBlock, [2, 2, 2, 2])  # Ensure this matches the saved model structure
    model.load_state_dict(torch.load(model_path))  # Load state_dict instead of full model
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to GPU if available
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to("cuda" if torch.cuda.is_available() else "cpu")
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    submission = pd.DataFrame({
        'ID': list(range(len(predictions))),
        'label': predictions
    })

    filename = f"submission-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    submission.to_csv(filename, index=False)
    return filename
    