import torch
import pandas as pd
import kaggle
import os
import zipfile
import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

os.environ['KAGGLE_USERNAME'] = 'hurryingauto3'
os.environ['KAGGLE_KEY'] = 'e52e89859abc30f3f9901784a3004779'

def download_and_prepare_data():
    
    kaggle.api.competition_download_files('deep-learning-spring-2025-project-1', path='.', unzip=True)

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = datasets.ImageFolder(root='./test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return test_loader

def generate_kaggle_predictions(model_path):
    test_loader = download_and_prepare_data()
    
    model = torch.load(model_path)
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'label': predictions
    })

    filename = f'submission-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.csv'
    submission.to_csv(filename, index=False)
    return filename

def submit_to_kaggle(filename, message):
    kaggle.api.competition_submit(filename, message, 'deep-learning-spring-2025-project-1')

model_path = 'model.pth'
message = 'Deep Learning Project 1 Accuracy Submission'

filename = generate_kaggle_predictions(model_path)
submit_to_kaggle(filename, message)