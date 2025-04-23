"""Initialize, and train, test and validate model on galaxy10 dataset."""
import argparse

import pytorch_lightning as pl
import torch

from galaxy_data import Galaxy10DataModule
from models import ViTClassifierHead, ViTBackbone, BaseViTLitModule
from callbacks import LearningRateMonitor

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning


torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore", category=PossibleUserWarning)


MAX_EPOCHS = 100
OVERFIT_BATCHES = 0.0
LIMIT_TRAIN_BATCHES = None
LIMIT_VAL_BATCHES = None
LIMIT_TEST_BATCHES = None
PRECISION = "32-true"
TEST = False
CHECK_VAL_EVERY_N_EPOCHS = 3


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
	trainer_group.add_argument("--limit_test_batches", type=float, default=LIMIT_TEST_BATCHES)
	trainer_group.add_argument("--limit_train_batches", type=float, default=LIMIT_TRAIN_BATCHES)
	trainer_group.add_argument("--precision", type=str, default=PRECISION)
	trainer_group.add_argument("--test", action="store_true", default=TEST)
	trainer_group.add_argument("--check_val_every_n_epochs", type=int, default=CHECK_VAL_EVERY_N_EPOCHS)

	# Add data and model specific arguments
	data_group = parser.add_argument_group("Data Args")
	Galaxy10DataModule.add_to_argparse(data_group)

	model_group = parser.add_argument_group("Model Args")
	ViTClassifierHead.add_to_argparse(model_group)
	ViTBackbone.add_to_argparse(model_group)

	lit_model_group = parser.add_argument_group("LitModel Args")
	BaseViTLitModule.add_to_argparse(lit_model_group)

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
	backbone_model = ViTBackbone(args=args)
	head_model = ViTClassifierHead(backbone_config=backbone_model.config, args=args)

	if args.load_checkpoint is not None:
		lit_model = BaseViTLitModule.load_from_checkpoint(
			args.load_checkpoint, 
			args=args, 
			head_model=head_model, 
			backbone_model=backbone_model
		)
		print(f"Model successfully loaded from checkpoint {args.load_checkpoint}")
	else: 
		lit_model = BaseViTLitModule(args=args, head_model=head_model, backbone_model=backbone_model)

	# Load LightningDataModule
	data = Galaxy10DataModule(args=args)

	# Define custom callbacks for logging
	filename_format = "epoch={epoch:04d}-val.loss={val/loss:.3f}"
	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		save_top_k=3,
		monitor="val/loss", 
		filename=filename_format,
		mode="min",
		auto_insert_metric_name=False,
		every_n_epochs=args.check_val_every_n_epochs 
	)
	summary_callback = pl.callbacks.ModelSummary(max_depth=2)
	callbacks = [checkpoint_callback, summary_callback, LearningRateMonitor()]

	# Initialize trainer
	trainer = pl.Trainer(
		max_epochs=args.max_epochs, 
		overfit_batches=args.overfit_batches, 
		limit_val_batches=args.limit_val_batches, 
		limit_test_batches=args.limit_test_batches,
		limit_train_batches=args.limit_train_batches,
		precision=args.precision, 
		callbacks=callbacks
	)

	trainer.fit(lit_model, datamodule=data)
	
	if args.test:
		trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
	main()
