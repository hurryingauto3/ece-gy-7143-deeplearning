# Model Definition for ResNet Architecture
import torch
import datetime
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary


def get_hyperparameters():
    """Returns hyperparameters for training the model."""
    return {
        "lr": 0.001,
        "weight_decay": 5e-4,
        "batch_size": 128,
    }


def get_scheduler(optimizer, scheduler_type, T_max=200):
    """Returns the learning rate scheduler for the optimizer."""

    def warmup_lr(step):
        return min(1.0, step / 5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    if scheduler_type == "warmup":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lr, scheduler)
    return scheduler


def get_optimizer(model, lr, weight_decay, momentum=0.9, type_="sgd"):
    optimizer_dict = {
        "SGD": optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        ),
        "Adam": optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        "RMSProp": optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay),
        "Adafactor": optim.Adafactor(
            model.parameters(), lr=lr, weight_decay=weight_decay
        ),
    }
    return optimizer_dict[type_]


def get_critereon(label_smoothing=None):
    """Returns the loss function for training the model."""
    return (
        torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        if label_smoothing
        else torch.nn.CrossEntropyLoss()
    )


def train(
    model, train_loader, test_loader, device, optimizer, criterion, scheduler=None
):
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
    epoch_train_losses = []  # Average training loss per epoch
    epoch_test_losses = []  # Average test loss per epoch

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
        if scheduler:
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

        print(
            f"Epoch {epoch}, Accuracy: {accuracy:.2f}%, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        )

        # Early stopping if accuracy is high enough
        if accuracy > 88:
            final_accuracy = accuracy
            break

    print(f"Final Accuracy: {final_accuracy:.2f}%")
    return epoch_train_losses, epoch_test_losses


def plot_training_curves(train_losses, test_losses):
    """
    Plot the training and test losses.

    Args:
        train_losses (list): List of training losses
        test_losses (list): List of test losses
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(test_losses, label="Test Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.show()


def save_model(model, final_accuracy):
    """Saves the model with a timestamp and final accuracy.

    Args:
      model: The trained PyTorch model.
      final_accuracy: The final accuracy achieved by the model.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"model-{timestamp}-{final_accuracy:.2f}.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")
