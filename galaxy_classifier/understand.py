import torch
import matplotlib.pyplot as plt
from astroNN.datasets import galaxy10
from torch import nn
from torch.utils import data


def inspect_predictions(
    model: nn.Module,
    data_loader: data.DataLoader,
    batch_size: int = 16,
    batches: int = 1
) -> None:
    """Visualizes model predictions versus ground truth to 
    evaluate model performance. This will help with assessing
    whether model prediction errors reveal systematic issues,
    or reflect inherent difficulty in assigning class labels 
    to ambuguous samples.

    Args:
        model:
            Model to assess.
        data_loader:
            Instance of PyTorch DataLoader class.
        batch_size: 
            Number of samples in single batch of DataLoader 
            instance.
        batches:
            Number of batches from provided DataLoader to assess.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    fig_dim = 8
    n_columns = 4
    n_rows = batch_size * batches // n_columns

    plt.figure(figsize=(fig_dim, fig_dim*batches))

    i = 1
    for idx, batch in enumerate(data_loader):
        if idx >= batches:
            break
        images, labels = batch["image"], batch["label"]
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)

        for image, label, pred in zip(images, labels, preds):
            true_label = galaxy10.galaxy10cls_lookup(label.item())
            pred_label = galaxy10.galaxy10cls_lookup(pred.item())
            plt.subplot(n_rows, n_columns, i)
            plt.imshow(image.cpu().permute(1, 2, 0))
            plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=8)
            plt.axis("off")
            i += 1

    plt.tight_layout()
    plt.show()