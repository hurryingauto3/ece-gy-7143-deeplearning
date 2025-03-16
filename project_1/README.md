# CIFAR-10 Classification Notebook

## Overview
- Trains a ResNet variant with Squeeze-and-Excitation blocks on CIFAR-10.
- Uses advanced augmentations (MixUp, CutMix) and SWA for improved generalization.
- Integrates Kaggle API for downloading competition data and generating submission files.
- All models are saved with a timestamp and accuracy for reproducibility and can be found in the `models` directory.

## Features
- **Data Module:** Custom DataLoader for standard, training, and competition test sets.
- **Model:** Custom ResNet with SE blocks, reduced channel sizes, and drop path regularization.
- **Augmentation:** Heavy/light transforms, MixUp, and CutMix.
- **Training:** Incorporates SWA, learning rate scheduling, and batch normalization freezing.
- **Output:** Plots training curves, saves model checkpoints, and produces Kaggle submission CSV.

## Requirements
- Python 3.8+
- PyTorch & Torchvision
- NumPy, Pandas, Matplotlib
- Kaggle API
- tqdm, torchsummary

## Setup
1. **Kaggle Credentials:** Notebook automatically creates `~/.config/kaggle/kaggle.json`.
2. **Dataset:** Downloads CIFAR-10 and competition test data if not present.
3. **Install Dependencies:**
   ```bash
   pip install torch torchvision numpy pandas matplotlib kaggle tqdm torchsummary

## Usage
- Display Data: Shows batches from training, test, and competition sets.
- Training: Modify hyperparameters in the training loop. Run the notebook or script to train.
- Plotting: Generates training/test loss curves.
- Model Saving: Saves the trained model with timestamp and accuracy.
- Kaggle Submission: Generates a CSV submission file using the Kaggle API.

## Running
Open in Jupyter Notebook:
```bash
jupyter notebook ece_7123_project_1_final.ipynb
```
Or run on Kaggle:
- https://www.kaggle.com/code/hurryingauto3/notebookeb139646e3


## References
- CIFAR-10 Dataset
- Kaggle Competition
- PyTorch
- Kaggle API GitHub

## URLs:
- https://www.cs.toronto.edu/~kriz/cifar.html  
- https://www.kaggle.com/competitions/deep-learning-spring-2025-project-1  
- https://pytorch.org/  
- https://github.com/Kaggle/kaggle-api