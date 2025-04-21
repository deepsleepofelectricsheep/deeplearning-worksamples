"""PyTorch NN models for galaxy classification task."""
import argparse

import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy
from transformers import ViTModel

import pytorch_lightning as pl


NUM_CLASSES = 10
LR = 3e-4
VIT_PRETRAINED_MODEL_NAME = "facebook/dino-vits16"
	

class DinoLinearClassifier(torch.nn.Module):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        num_classes = self.args.get("num_classes", NUM_CLASSES)
        vit_pretrained_model_name = self.args.get("vit_pretrained_model_name", VIT_PRETRAINED_MODEL_NAME)
        self.feature_extractor = ViTModel.from_pretrained(vit_pretrained_model_name)
        self.linear = torch.nn.Linear(self.feature_extractor.config.hidden_size, num_classes)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)["last_hidden_state"][:, 0]
        output = self.linear(features)
        return output

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
        parser.add_argument("--vit_pretrained_model_name", type=str, default=VIT_PRETRAINED_MODEL_NAME)
        return parser
	

class LitModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}
        num_classes = self.args.get("num_classes", NUM_CLASSES)
        self.lr = self.args.get("lr", LR)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy_fn = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        inputs, targets = batch
        outs = self(inputs)
        loss = self.loss_fn(outs, targets)
        accuracy = self.accuracy_fn(outs, targets)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log(f"{stage}_accuracy", accuracy, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")
    
    def testing_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--lr", type=float, default=LR)
        return parser