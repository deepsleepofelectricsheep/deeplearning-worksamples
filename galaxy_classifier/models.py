"""PyTorch NN models for galaxy classification task."""
import argparse

import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy
from transformers import ViTModel
from transformers.models.vit.configuration_vit import ViTConfig

import pytorch_lightning as pl


NUM_CLASSES = 10
LR = 3e-4
VIT_PRETRAINED_MODEL_NAME = "facebook/dino-vits16"
LINEAR_CLASSIFIER_HIDDEN_SIZE = 512
HEAD_TYPE = "linear"
HIDDEN_SIZE = 128
P_DROPOUT = 0.1
FREEZE_BACKBONE = True
RETURN_MEAN_TOKEN_EMBEDDINGS = False


class ViTClassifierHead(torch.nn.Module):
    def __init__(self, backbone_config: dict, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.num_classes = self.args.get("num_classes", NUM_CLASSES)
        self.head_type = self.args.get("head_type", HEAD_TYPE)
        self.hidden_size = self.args.get("hidden_size", HIDDEN_SIZE)
        self.p_dropout = self.args.get("p_dropout", P_DROPOUT)
        self.input_size = backbone_config["hidden_size"]
        
        if self.head_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.p_dropout),
                nn.Linear(self.hidden_size, self.num_classes)
            )
        else:
            self.model = nn.Linear(self.input_size, self.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
    
    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
        parser.add_argument("--head_type", type=str, default=HEAD_TYPE)
        parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
        parser.add_argument("--p_dropout", type=float, default=P_DROPOUT)
        return parser
    

class ViTBackbone(nn.Module):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.vit_pretrained_model_name = self.args.get("vit_pretrained_model_name", VIT_PRETRAINED_MODEL_NAME)
        self.freeze_backbone = self.args.get("freeze_backbone", FREEZE_BACKBONE)
        self.return_mean_token_embeddings = self.args.get("return_mean_token_embeddings", RETURN_MEAN_TOKEN_EMBEDDINGS)
        self.model = ViTModel.from_pretrained(self.vit_pretrained_model_name)
        self.config = self.model.config.to_dict()

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)["last_hidden_state"]
        outputs = outputs.mean(dim=1) if self.return_mean_token_embeddings else outputs[:, 0]
        return outputs
    
    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--vit_pretrained_model_name", type=str, default=VIT_PRETRAINED_MODEL_NAME)
        parser.add_argument("--freeze_backbone", type=bool, default=FREEZE_BACKBONE)
        parser.add_argument("--return_mean_token_embeddings", type=bool, default=RETURN_MEAN_TOKEN_EMBEDDINGS)
        return parser
    

class BaseViTLitModule(pl.LightningModule):
    def __init__(self, head_model: nn.Module, backbone_model: nn.Module, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.head_model = head_model
        self.backbone_model = backbone_model
        self.args = vars(args) if args is not None else {}
        self.num_classes = self.args.get("num_classes", NUM_CLASSES)
        self.lr = self.args.get("lr", LR)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy_fn = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone_model(x)
        outputs = self.head_model(features)
        return outputs
    
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
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--lr", type=float, default=LR)
        return parser