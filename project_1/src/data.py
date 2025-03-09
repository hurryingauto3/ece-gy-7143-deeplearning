#!/usr/bin/env python
"""
Data handling module for CIFAR-10 dataset and competition data.
"""
import os
import zipfile
import pickle
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

class CIFARTransforms:
    def __init__(self, mode):
        self.mode = mode
        if mode == "basic":
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif mode == "advanced":
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),  # Diverse randomized policies
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Occlude parts of image
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise ValueError("Invalid mode. Choose 'basic' or 'advanced'.")


class CIFARTestDataset(Dataset):
    """
    Custom dataset for CIFAR-10 test images.

    Args:
        pkl_file (str): Path to the pickle file containing the test data.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, pkl_file, transform=None):
        with open(pkl_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        self.images = data_dict[b'data'] if b'data' in data_dict else data_dict['data']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        # Check if the image is already in (32,32,3) format.
        if img.ndim == 3 and img.shape[2] == 3:
            # Already in correct shape, ensure type is uint8.
            pil_img = Image.fromarray(img.astype('uint8'))
        else:
            # Otherwise assume it's flat and reshape.
            pil_img = Image.fromarray(img.reshape(3, 32, 32).transpose(1, 2, 0).astype('uint8'))

        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, index


class CIFAR10DataModule:
    """
    Data module for CIFAR10 and competition test data.

    Args:
        data_dir (str): Directory to store data.
        competition_name (str): Name of the Kaggle competition.
        batch_size (int): Training batch size.
        test_batch_size (int): Testing batch size.
        num_workers (int): Number of workers for data loading.
    """
    def __init__(self, transform_train, transform_test, data_dir="./data", competition_name="deep-learning-spring-2025-project-1",
                batch_size=128, test_batch_size=100, num_workers=2):
        self.data_dir = data_dir
        self.competition_name = competition_name
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        # Paths for competition data
        self.competition_path = os.path.join(self.data_dir, self.competition_name)
        self.zip_path = os.path.join(self.competition_path, f"{self.competition_name}.zip")
        self.test_pkl = os.path.join(self.competition_path, "cifar_test_nolabel.pkl")

        # Standard transforms for CIFAR10
        self.transform_train = transform_train
        self.transform_test = transform_test

    def get_train_loader(self):
        train_dataset = datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.transform_train
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def get_standard_test_loader(self):
        test_dataset = datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.transform_test
        )
        return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def download_competition_data(self):
        if not os.path.exists(self.test_pkl):
            os.makedirs(self.competition_path, exist_ok=True)
            api = KaggleApi()
            api.authenticate()
            api.competition_download_files(self.competition_name, path=self.competition_path)
            if os.path.exists(self.zip_path):
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.competition_path)
        if not os.path.exists(self.test_pkl):
            raise FileNotFoundError(f"Competition test file not found in '{self.test_pkl}'")

    def get_competition_test_loader(self):
        self.download_competition_data()
        test_dataset = CIFARTestDataset(self.test_pkl, transform=self.transform_test)
        return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers)

# Helper functions for displaying images
def imshow(img, title=None):
    """
    Unnormalizes and displays an image grid.
    """
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = std * img + mean  # unnormalize
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

def display_batch(loader, nrow=16, title="Data Batch"):
    """
    Retrieves one batch from the provided DataLoader, displays the image grid,
    and prints the corresponding labels.
    """
    images, labels = next(iter(loader))
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    imshow(grid, title=title)
    print(f"{title} Labels:\n", labels)

def display_all_data(train_loader, test_loader, competition_loader):
    """
    Displays a batch of images and labels for training data, standard test data,
    and competition test data.
    """
    display_batch(train_loader, nrow=16, title="Training Data")
    display_batch(test_loader, nrow=16, title="Standard Test Data")
    display_batch(competition_loader, nrow=16, title="Competition Test Data")

def generate_submission(model, test_loader, filename_suffix="", device=None):
    """
    Generates a submission CSV file from predictions on the competition test dataset.

    Parameters:
        model: The trained PyTorch model.
        test_loader: DataLoader for the competition test dataset.
        filename_suffix (str): Optional suffix to add to the filename.
        device: Torch device to use (defaults to CUDA if available).

    Returns:
        submission (pd.DataFrame): DataFrame containing "ID" and "Labels".
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, indices in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)  # Get predicted labels (0-9)
            all_ids.extend(indices.tolist())
            all_preds.extend(preds.cpu().tolist())

    submission = pd.DataFrame({"ID": all_ids, "Labels": all_preds})
    filename = f"submission-{filename_suffix}.csv" if filename_suffix else "submission.csv"
    submission.to_csv(filename, index=False)

    print(f"Submission file saved as {filename}")
    return submission

def plot_random_competition_predictions(model, competition_test_loader, num_images=3):
    """
    Plots a random selection of images from the competition test dataset along with their predicted labels.

    Parameters:
        model: A trained PyTorch model.
        competition_test_loader: A DataLoader for the competition test dataset.
        num_images (int): Number of random images to display.
    """
    # Set up device and model evaluation mode
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    # Get one batch from the competition test loader
    images, _ = next(iter(competition_test_loader))
    images = images.to(device)

    # Predict labels using the model
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)  # Get integer labels 0-9

    # Define a function to unnormalize images
    def unnormalize(img_tensor):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        return img_tensor * std + mean

    # Unnormalize and convert images to numpy format for plotting
    images_disp = unnormalize(images.cpu())
    images_np = images_disp.numpy().transpose(0, 2, 3, 1)

    # Randomly select indices from the batch
    perm = torch.randperm(images.size(0))
    selected = perm[:num_images]

    # Plot the selected images with their predicted labels
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(selected):
        ax = plt.subplot(1, num_images, i + 1)
        plt.imshow(np.clip(images_np[idx], 0, 1))
        plt.title(f"Predicted: {preds[idx].item()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
