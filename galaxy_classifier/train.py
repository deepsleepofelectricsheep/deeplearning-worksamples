"""Initialize, and train, test and validate model on galaxy10 dataset."""
import argparse

import pytorch_lightning as pl
import torch

from galaxy_data import Galaxy10DataModule
from models import LitModule, DinoLinearClassifier

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning


warnings.filterwarnings("ignore", category=PossibleUserWarning)


MAX_EPOCHS = 100
OVERFIT_BATCHES = 1
LIMIT_VAL_BATCHES = 0.0
PRECISION = "32-true"


def _setup_parser():
	"""Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
	parser = argparse.ArgumentParser(add_help=False)

	# Add basic arguments
	parser.add_argument("--load_checkpoint", type=str, default=None, help="If passed, loads a model from the provided path.")	

	# Add Trainer specific arguments, such as --max_epochs, --overfit_batches, --limit_val_batches and --precision
	trainer_group = parser.add_argument_group("Trainer Args")
	trainer_group.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
	trainer_group.add_argument("--overfit_batches", type=float, default=OVERFIT_BATCHES)
	trainer_group.add_argument("--limit_val_batches", type=float, default=LIMIT_VAL_BATCHES)
	trainer_group.add_argument("--precision", type=str, default=PRECISION)

	# Add data and model specific arguments
	data_group = parser.add_argument_group("Data Args")
	Galaxy10DataModule.add_to_argparse(data_group)

	model_group = parser.add_argument_group("Model Args")
	DinoLinearClassifier.add_to_argparse(model_group)

	lit_model_group = parser.add_argument_group("LitModel Args")
	LitModule.add_to_argparse(lit_model_group)

	parser.add_argument("--help", "-h", action="help")
	return parser


def main():
	"""
	Run an experiment.
	
	Sample command:
	```
	python train.py --overfit_batches 1 --batch_size 8 --max_epochs 10
	```

	For basic help documentation, run the command
    ```
    python training/run_experiment.py --help
    ```
	"""
	parser = _setup_parser()
	args = parser.parse_args()

	# Load LightningModule
	model = DinoLinearClassifier(args=args)
	if args.load_checkpoint is not None:
		lit_model = LitModule.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
	else: 
		lit_model = LitModule(args=args, model=model)

	# Load LightningDataModule
	data = Galaxy10DataModule(args=args)

	# Initialize trainer
	trainer = pl.Trainer(
		max_epochs=args.max_epochs, 
		overfit_batches=args.overfit_batches, 
		limit_val_batches=args.limit_val_batches, 
		precision=args.precision
	)

	trainer.fit(lit_model, datamodule=data)


if __name__ == "__main__":
	main()
