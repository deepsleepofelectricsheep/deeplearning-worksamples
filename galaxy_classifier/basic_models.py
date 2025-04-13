"""PyTorch NN models for galaxy classification task."""
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy

import pytorch_lightning as pl


class MLP(nn.Module):
	"""Simple MLP to evaluate classification performance baseline."""

	def __init__(
		self,
		channels: int = 3,
		width: int = 256,
		height: int = 256,
		num_classes: int = 10,
		hidden_size: int = 64
    ) -> None:
		super().__init__()
		self.model = nn.Sequential(
			nn.Flatten(),
			nn.Linear(channels * width * height, hidden_size),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(hidden_size, num_classes)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)


class CNNBlock(nn.Module):
	"""Reusable CNN block with Conv2d -> ReLU -> MaxPool2d."""

	def __init__(
		self, 
		in_channels: int, 
		out_channels: int, 
		kernel_size: int = 3, 
		pool_size: int = 2
	) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=pool_size)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class CNN(nn.Module):
	"""Simple four layer CNN model to establish classification performance baseline."""

	def __init__(
		self,
		block_one_in_channels: int = 3,
		block_one_out_channels: int = 32,
		block_two_in_channels: int = 32,
		block_two_out_channels: int = 64,
		block_three_in_channels: int = 64,
		block_three_out_channels: int = 128,
		block_four_in_channels: int = 128,
		block_four_out_channels: int = 256,
		block_kernel_size: int = 3,
		block_pool_size: int = 2,
		avg_pool_size: int = 4,
		num_classes: int = 10
	) -> None:
		super().__init__()
		self.block_one = CNNBlock(
			in_channels=block_one_in_channels, 
			out_channels=block_one_out_channels,
			kernel_size=block_kernel_size,
			pool_size=block_pool_size
		)
		self.block_two = CNNBlock(
			in_channels=block_two_in_channels,
			out_channels=block_two_out_channels,
			kernel_size=block_kernel_size,
			pool_size=block_pool_size
		)
		self.block_three = CNNBlock(
			in_channels=block_three_in_channels,
			out_channels=block_three_out_channels,
			kernel_size=block_kernel_size,
			pool_size=block_pool_size
		)
		self.block_four = CNNBlock(
			in_channels=block_four_in_channels,
			out_channels=block_four_out_channels,
			kernel_size=block_kernel_size,
			pool_size=block_pool_size
		)
		self.pool = nn.AdaptiveAvgPool2d((avg_pool_size, avg_pool_size))
		self.flatten = nn.Flatten()
		self.classification_layer = nn.Linear(
			block_four_out_channels * avg_pool_size * avg_pool_size, 
			num_classes
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.block_one(x)
		out = self.block_two(out)
		out = self.block_three(out)
		out = self.block_four(out)
		out = self.pool(out)
		out = self.flatten(out)
		out = self.classification_layer(out)

		return out


class LitModel(pl.LightningModule):
	"""Generic PyTorch-Lightning class that must be initialized with a PyTorch Module."""

	def __init__(self, model: nn.Module,num_classes: int = 10, lr: float = 1e-4) -> None:
		super().__init__()
		self.model = model
		self.num_classes = num_classes
		self.lr = lr
		self.loss_fn = nn.CrossEntropyLoss()
		self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
		return optimizer

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)

	def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
		xb, yb = batch
		logits = self(xb)
		loss = self.loss_fn(logits, yb)
		acc = self.accuracy(logits, yb)
		self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
		self.log(f"{stage}_accuracy", acc, on_epoch=True, on_step=False, prog_bar=True)
		return loss 

	def training_step(
		self,
		batch: torch.Tensor,
		batch_idx: int
	) -> torch.Tensor:
		return self._shared_step(batch, "train")

	def validation_step(
		self,
		batch: torch.Tensor,
		batch_idx: int
	) -> torch.Tensor:
		return self._shared_step(batch, "val")

	def test_step(
		self,
		batch: torch.Tensor,
		batch_idx: int
	) -> torch.Tensor:
		return self._shared_step(batch, "test")	