"""Initialize, and train, test and validate model on galaxy10 dataset."""
import argparse

import pytorch_lightning as pl
import torch

from data import _download_split_and_load_data
from basic_models import MLP, CNN, LitModel

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning


warnings.filterwarnings("ignore", category=PossibleUserWarning)


BATCH_SIZE = 8
MAX_EPOCHS = 5
DEFAULT_MODEL_CLASS = "CNN"
MODEL_CLASSES = {
	"CNN": CNN,
	"MLP": MLP
}
OVERFIT_BATCHES = 0
LIMIT_VAL_BATCHES = 0.0


def main():
	"""
	Run an experiment.
	
	Sample command:
	```
	python train.py --overfit_batches 1 --batch_size 8 --max_epochs 10
	```
	"""
	
	# Setup Python's ArgumentParser
	parser = argparse.ArgumentParser()

	# Basic arguments
	parser.add_argument(
		"--overfit_batches",
		type=int,
		default=OVERFIT_BATCHES
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=BATCH_SIZE
	)
	parser.add_argument(
		"--max_epochs",
		type=int,
		default=MAX_EPOCHS
	)
	parser.add_argument(
		"--limit_val_batches",
		type=float,
		default=LIMIT_VAL_BATCHES
	)
	parser.add_argument(
		"--model_class",
		type=str,
		default=DEFAULT_MODEL_CLASS
	)

	# Parse arguments
	args = parser.parse_args()
	overfit_batches = args.overfit_batches
	batch_size = args.batch_size
	max_epochs = args.max_epochs
	limit_val_batches = args.limit_val_batches
	model_class = MODEL_CLASSES[args.model_class]

	# Load data into DataLoader
	train_dl, val_dl, test_dl = _download_split_and_load_data()

	# Initialize model
	model = model_class() # we can just use the defaults here
	lit_model = LitModel(model)

	# Tradeoff precision for performance
	torch.set_float32_matmul_precision("medium")

	# Initialize trainer
	trainer = pl.Trainer(
		max_epochs=max_epochs,
		overfit_batches=overfit_batches,
		limit_val_batches=limit_val_batches
	)

	# Fit model using trainer
	trainer.fit(lit_model, train_dl, val_dl)

if __name__ == "__main__":
	main()
