"""
Data module for the Galaxy10 classification task.

Here we will define a PyTorch Dataset object, and a function that
downloads the Galaxy10 dataset, and returns a PyTorch DataLoader 
object. 

Reference: https://astronn.readthedocs.io/en/latest/galaxy10.html
"""
import argparse
import random
from typing import Union

import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, Resize
from transformers import ViTImageProcessor

from astroNN.datasets import load_galaxy10

from PIL.Image import fromarray


BATCH_SIZE = 8
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
CROP_SIZE = 224
VIT_PRETRAINED_MODEL_NAME = "facebook/dino-vits16"


class Galaxy10Dataset(Dataset):
    """PyTorch Dataset object that transforms and loads the data."""
    def __init__(
        self,
        images: list,
        labels: list,
        stage: str = "train",
        transform: Union[Compose, ViTImageProcessor] = None,
        crop_size: int = CROP_SIZE,
    ) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.stage = stage
        self.transform = transform if transform else Compose([Resize((crop_size, crop_size)), ToTensor()])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = fromarray(self.images[idx].astype("uint8"))
        label = self.labels[idx]

        if self.stage == "train":
            image = RandomHorizontalFlip()(image)

        if isinstance(self.transform, ViTImageProcessor):
            image = self.transform(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.labels)


class Galaxy10DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            args: argparse.Namespace = None, 
            train_transform: Union[Compose, ViTImageProcessor] = None, 
            test_transform: Union[Compose, ViTImageProcessor] = None, 
            val_transform: Union[Compose, ViTImageProcessor] = None
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.test_split = self.args.get("test_split", TEST_SPLIT)
        self.val_split = self.args.get("val_split", VAL_SPLIT)
        vit_pretrained_model_name = self.args.get("vit_pretrained_model_name", VIT_PRETRAINED_MODEL_NAME)

        # Default ViTImageProcessor can be overwritten by passing transformations as arguments
        self.train_transform = train_transform if train_transform is not None else ViTImageProcessor.from_pretrained(vit_pretrained_model_name)
        self.test_transform = test_transform if test_transform is not None else ViTImageProcessor.from_pretrained(vit_pretrained_model_name)
        self.val_transform = val_transform if val_transform is not None else ViTImageProcessor.from_pretrained(vit_pretrained_model_name)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--test_split", type=float, default=TEST_SPLIT)
        parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
        parser.add_argument("--vit_pretrained_model_name", type=str, default=VIT_PRETRAINED_MODEL_NAME)

    def setup(self, stage: str):
        # Download the data
        images, labels = load_galaxy10()

        # Shuffle data before splitting
        tmp = list(zip(images, labels))
        random.shuffle(tmp)
        images, labels = zip(*tmp)
        del tmp  # free memory

        total_size = len(images)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - test_size - val_size

        train_data = [images[:train_size], labels[:train_size]]
        val_data = [images[train_size:train_size + val_size], labels[train_size:train_size + val_size]]
        test_data = [images[train_size + val_size:], labels[train_size + val_size:]]

        del images, labels  # free memory

        self.train_ds = Galaxy10Dataset(*train_data, transform=self.train_transform, stage="train")
        self.val_ds = Galaxy10Dataset(*val_data, transform=self.val_transform, stage="val")
        self.test_ds = Galaxy10Dataset(*test_data, transform=self.test_transform, stage="test")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)