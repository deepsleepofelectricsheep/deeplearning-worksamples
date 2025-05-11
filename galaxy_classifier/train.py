import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from typing import Callable
from torch.utils import data
from torch.nn import utils
from torchvision.transforms import v2


def train(
    model: Callable[[torch.Tensor], torch.Tensor], 
    data_loaders: dict[str, data.DataLoader],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler, 
    n_epochs: int = 5,
    batch_size: int = 16,
    gradient_clip: float = 1.0,
    num_classes: int = 10
) -> dict[str, list]:
    """Trains provided model for specified number of epochs, on provided 
    train and val dataloaders, given provided loss function and optimizer.

    Args:
        model: 
            Provided classification model.
        n_epochs: 
            Number of epochs.
        loss_fn: 
            Loss function that takes two tensors as inputs, and returns
            a tensor.
        data_loaders: 
            Dictionary with values 'train' and 'val' with corresponding 
            data loaders.
        optimizer: 
            PyTorch Optimizer class instance.
        scheduler:
            PyTorch _LRScheduler instance.
        batch_size: 
            Number of samples in a batch in the train and val data 
            loaders.
        gradient_clip:
            Parameter to limit magnitude of gradients in backprop to prevent
            exploding gradients.
        num_classes:
            Number of classes in the dataset.
    """
    history = {
        "train_loss": [0] * n_epochs,
        "train_accuracy": [0] * n_epochs,
        "val_loss": [0] * n_epochs,
        "val_accuracy": [0] * n_epochs
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    mixup = v2.MixUp(num_classes=num_classes, alpha=0.8) # Initialize Mixup augmentation

    for epoch in range(n_epochs):
        model.train()
        for batch in data_loaders["train"]:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device) 
            images, labels = mixup(images, labels.long()) # Augment data
            pred = model(images)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip is not None:
                utils.clip_grad_norm_(model.parameters(), gradient_clip) # Add gradient clip
            optimizer.step()

            is_correct = (torch.argmax(pred, dim=1) == torch.argmax(labels, dim=1)).float().mean()
            history["train_accuracy"][epoch] += is_correct.item()
            history["train_loss"][epoch] += loss.item()

        scheduler.step()

        history["train_accuracy"][epoch] /= len(data_loaders["train"].dataset)/batch_size
        history["train_loss"][epoch] /= len(data_loaders["train"].dataset)/batch_size

        print(
            f"Epoch [{epoch+1}/{n_epochs}]: "
            f"Train Loss = {history['train_loss'][epoch]:0.4f}; "
            f"Train Acc. = {history['train_accuracy'][epoch]:0.4f}"
        )

        model.eval()
        with torch.no_grad():
            for batch in data_loaders["val"]:
                images, labels = batch["image"], batch["label"]
                images, labels = images.to(device), labels.to(device)
                pred = model(images)
                loss = loss_fn(pred, labels)

                is_correct = (torch.argmax(pred, dim=1) == labels).float().mean()
                history["val_accuracy"][epoch] += is_correct.item()
                history["val_loss"][epoch] += loss.item()  

        history["val_accuracy"][epoch] /= len(data_loaders["val"].dataset)/batch_size
        history["val_loss"][epoch] /= len(data_loaders["val"].dataset)/batch_size

        print(
            f"Epoch [{epoch+1}/{n_epochs}]: "
            f"Val Loss = {history['val_loss'][epoch]:0.4f}; "
            f"Val Acc. = {history['val_accuracy'][epoch]:0.4f}"
        )

        print()

    return history


def predict(
    model: nn.Module,
    data_loader: data.DataLoader
) -> tuple[list, list]:
    """Generates predictions for all samples in provided
    dataloader.

    Args:
        model:
            PyTorch model to generate predictions
        data_loader:
            PyTorch DataLoader instance with samples for
            prediction generation.

    Returns:
        Tuple consisting of two lists, the predicted, and 
        the ground truth labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y = []
    y_hat = []

    model.eval()
    for batch in data_loader:
        images, labels = batch["image"], batch["label"]
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)

        y.extend(labels.tolist())
        y_hat.extend(preds.tolist())

    return y_hat, y
