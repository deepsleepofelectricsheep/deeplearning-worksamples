"""
Data module for the Galaxy10 classification task.

Here we will define a PyTorch Lightning DataModule that downloads,
preprocesses, and splits the data.

Reference: https://astronn.readthedocs.io/en/latest/galaxy10.html
"""
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import numpy as np

import torch 
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
import pytorch_lightning as pl 

from astroNN.datasets import load_galaxy10


SequenceOrTensor = Union[Sequence, torch.Tensor]


class Galaxy10Dataset(Dataset):
	"""PyTorch Dataset object that transforms and loads the data."""

	def __init__(
		self, 
		images: SequenceOrTensor, 
		labels: SequenceOrTensor,
		transform: Callable = None,
	) -> None:
		self.images = images
		self.labels = labels
		self.transform = transform

	def __len__(self) -> int:
		return len(self.images)

	def __getitem__(
		self, 
		idx: int
	) -> tuple[torch.Tensor, torch.Tensor]:
		image = self.images[idx]
		label = self.labels[idx]

		if self.transform:
			image = self.transform(image)

		return image, label


class Galaxy10DataModule(pl.LightningDataModule):
	"""LigntningDataModule to manage data in Pytorch Lightning."""

	def __init__(
		self,
		batch_size: int = 8,
		val_split: float = 0.1,
		test_split: float = 0.1,
		seed: int = 42,
		num_workers: int = 0,
		train_transform: Callable = None, 
		val_transform: Callable = None,
		test_transform: Callable = None,
	) -> None:
		super().__init__()
		self.batch_size = batch_size
		self.val_split = val_split
		self.test_split = test_split
		self.seed = seed
		self.num_workers = num_workers

		self.train_transform = train_transform or transforms.ToTensor()
		self.val_transform = val_transform or transforms.ToTensor()
		self.test_transform = test_transform or transforms.ToTensor()

		self.train_dataset = None
		self.val_dataset = None
		self.test_dataset = None

	def prepare_data(self) -> None:
		# Download the data, if needed
		_ = load_galaxy10()

	def setup(
		self,
		stage: str = None
	) -> None:
		# Load the data
		images, labels = load_galaxy10()
		images = torch.from_numpy(images).float() / 255.0
		labels = torch.from_numpy(labels).long()

		# Split the data
		total_size = len(images)
		val_size = int(total_size * self.val_split)
		test_size = int(total_size * self.test_split)
		train_size = total_size - val_size - test_size

		train_data, val_data, test_data = random_split(
			list(zip(images, labels)),
			lengths=[train_size, val_size, test_size],
			generator=torch.Generator().manual_seed(self.seed)
		)

		# Initialize train, test and valid dataset objects
		self.train_dataset = Galaxy10Dataset(
			images=torch.stack([x[0] for x in train_data]), 
			labels=torch.stack([x[1] for x in train_data]), 
			transform=self.train_transform
		)
		self.val_dataset = Galaxy10Dataset(
			images=torch.stack([x[0] for x in val_data]), 
			labels=torch.stack([x[1] for x in val_data]), 
			transform=self.val_transform
		)
		self.test_dataset = Galaxy10Dataset(
			images=torch.stack([x[0] for x in test_data]), 
			labels=torch.stack([x[1] for x in test_data]), 
			transform=self.test_transform
		)

	def train_dataloader(self) -> DataLoader:
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			pin_memory=True
		) 

	def val_dataloader(self) -> DataLoader:
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=True
		)

	def test_dataloader(self) -> DataLoader:
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=True
		)