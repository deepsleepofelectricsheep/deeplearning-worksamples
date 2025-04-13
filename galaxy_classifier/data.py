"""
Data module for the Galaxy10 classification task.

Here we will define a PyTorch Dataset object, and a function that
downloads the Galaxy10 dataset, and returns a PyTorch DataLoader 
object. 

Reference: https://astronn.readthedocs.io/en/latest/galaxy10.html
"""
import random

import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose

from astroNN.datasets import load_galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup

from PIL.Image import fromarray


BATCH_SIZE = 8
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1


class Galaxy10Dataset(Dataset):
    """PyTorch Dataset object that transforms and loads the data."""

    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        # Fallback to default if no transform is specified
        self.transform = transform if transform else Compose([ToTensor()])

    def __getitem__(self, idx):
        image = fromarray(self.images[idx].astype("uint8"))
        label = self.labels[idx]
        return self.transform(image), label

    def __len__(self):
        return len(self.labels)


def _download_split_and_load_data(
    batch_size=BATCH_SIZE, 
    test_split=TEST_SPLIT, 
    val_split=VAL_SPLIT,
    train_transform=None,
    test_transform=None, 
    val_transform=None
):
    """Helper function that downloads Galaxy10 dataset, performs train, test,
    and validation splitting, and loads the data into a PyTorch DataLoader.

    Returns
    -------
    torch.DataLoader
        Train DataLoader

    torch.DataLoader
        Valid DataLoader

    torch.DataLoader
        Test DataLoader
    """

    # Download the data
    ## Loads the images and labels data as numpy.ndarrays
    images, labels = load_galaxy10()

    # Shuffle data before splitting into training, validation and testing sets
    tmp = list(zip(images, labels))
    random.shuffle(tmp)
    images, labels = zip(*tmp)

    del tmp  # free up memory!

    total_size = len(images)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    train_data = [
        images[:train_size], 
        labels[:train_size]
    ]
    val_data = [
        images[train_size:train_size+val_size], 
        labels[train_size:train_size+val_size]
    ]
    test_data = [
        images[train_size+val_size:], 
        labels[train_size+val_size:]
    ]

    del images, labels  # free up memory!

    train_ds = Galaxy10Dataset(*train_data, train_transform)
    val_ds = Galaxy10Dataset(*val_data, val_transform)
    test_ds = Galaxy10Dataset(*test_data, test_transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, val_dl, test_dl
