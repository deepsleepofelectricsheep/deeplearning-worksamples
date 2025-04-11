"""torch nn modules  for galaxy classification task."""
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy

import pytorch_lightning as pl


class GalaxyMPL(pl.LightningModule):
	"""Simple MLP to evaluate classification performance baseline."""

	def __init__(
		self,
		input_size: int = 3*64*64,
		num_classes: int = 10,
		lr: float = 1e-3
	) -> None:
		super().__init__()
		self.save_hyperparameters()
		self.model = nn.Sequential(
			nn.Flatten(),
			nn.Linear(input_size, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes)
		)
		self.loss_fn = nn.CrossEntropyLoss()
		self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

	def forward(
		self,
		x: torch.Tensor,
	) -> torch.Tensor:
		"""Simple forward pass of MLP model."""
		return self.model(x)

	def _shared_step(
		self,
		batch: torch.Tensor, 
		stage: str
	):
		xb, yb = batch
		logits = self(xb)
		loss = self.loss_fn(logits, yb)
		acc = self.accuracy(logits, yb)
		self.log(f"{stage}_loss", loss, on_epoch=True)
		self.log(f"{stage}_accuracy", acc, on_epoch=True)
		return loss 

	def training_step(
		self,
		batch: torch.Tensor,
		batch_idx: int
	):
		return self._shared_step(batch, "train")

	def validation_step(
		self,
		batch: torch.Tensor,
		batch_idx: int
	):
		return self._shared_step(batch, "val")

	def test_step(
		self,
		batch: torch.Tensor,
		batch_idx: int
	):
		return self._shared_step(batch, "test")

	def configure_optimizers(self) -> torch.optim.Optimizer:
		optimizer = torch.optim.AdamW(
			self.parameters(), 
			lr=self.hparams.lr
		)
		return optimizer