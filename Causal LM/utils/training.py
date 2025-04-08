"""Download ebook(s) from project gutenberg, and train causal lm."""
import argparse
import requests
import pickle

from tokenizers import TokenEncoder
from dataset_modules import DatasetForCausalLM, load_data
from custom_causal_transformer import CausalTransformer, train, generate, configure_optimizer

import torch


EBOOK_LINKS = [
	"https://www.gutenberg.org/cache/epub/7849/pg7849.txt", 
	"https://www.gutenberg.org/cache/epub/5200/pg5200.txt"
]

TOKENIZER_SAVE_PATH = "tokenizers/token_encoder.pkl"

SEQ_LEN = 100
CHUNK_SIZE = 1
BATCH_SIZE = 8

D_MODEL = 512
NHEAD = 8
DIM_FEEDFORWARD = 2048
NUM_ENCODER_LAYERS = 6

LR = 1e-4

EPOCHS = 5
LOG_INTERVAL = 100

MODEL_SAVE_PATH = "models/sample_model.pt"


def _setup_parser():
	"""Set up Python's ArgumentParser."""
	
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument(
		"--ebook_link_1",
		type=str,
		default=EBOOK_LINKS[0],
		help="First of upto five Project Gutenberg ebook .txt links."
	)
	parser.add_argument(
		"--ebook_link_2",
		type=str,
		default=EBOOK_LINKS[1],
		help="Second of upto five Project Gutenberg ebook .txt links."

	)
	parser.add_argument(
		"--ebook_link_3",
		type=str,
		default=None,
		help="Third of upto five Project Gutenberg ebook .txt links."

	)
	parser.add_argument(
		"--ebook_link_4",
		type=str,
		default=None,
		help="Fourth of upto five Project Gutenberg ebook .txt links."

	)
	parser.add_argument(
		"--ebook_link_5",
		type=str,
		default=None,
		help="Fifth of upto five Project Gutenberg ebook .txt links."
	)
	parser.add_argument(
		"--seq_len",
		type=int,
		default=SEQ_LEN,
		help=("Maximum sequence length of input to train the causal lm" 
			  f"to predict the next token. Default: {SEQ_LEN}")
	)
	parser.add_argument(
		"--chunk_size",
		type=int,
		default=CHUNK_SIZE,
		help=f"Number of extra tokens to predict after input sequence. Default: {CHUNK_SIZE}"
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=BATCH_SIZE,
		help=f"Training batch size. Default: {BATCH_SIZE}"
	)
	parser.add_argument(
		"--d_model",
		type=int,
		default=D_MODEL,
		help=f"Dimension of embedding and model hidden size. Default: {D_MODEL}"
	)
	parser.add_argument(
		"--nhead",
		type=int,
		default=NHEAD,
		help=f"Number of attention heads. Default: {NHEAD}"
	)
	parser.add_argument(
		"--dim_feedforward",
		type=int,
		default=DIM_FEEDFORWARD,
		help=f"Dimension of feedforward layer in Transformer. Default: {DIM_FEEDFORWARD}"
	)
	parser.add_argument(
		"--num_encoder_layers",
		type=int,
		default=NUM_ENCODER_LAYERS,
		help=f"Number of encoder layers in the Transformer. Default: {NUM_ENCODER_LAYERS}"
	)
	parser.add_argument(
	"--epochs",
	type=int,
	default=EPOCHS,
	help=f"Number of training epochs. Default: {EPOCHS}"
	)
	parser.add_argument(
		"--log_interval",
		type=int,
		default=LOG_INTERVAL,
		help=f"How often to log training loss. Default: {LOG_INTERVAL}"
	)
	parser.add_argument(
		"--lr",
		type=int,
		default=LR,
		help=f"Learning rate for model optimier. Default: {LR}"
	)
	parser.add_argument(
		"--model_save_path",
		type=str,
		default=MODEL_SAVE_PATH,
		help=f"File path to save model. Default: {MODEL_SAVE_PATH}"
	)
	parser.add_argument(
		"--tokenizer_save_path",
		type=str,
		default=TOKENIZER_SAVE_PATH,
		help=f"File path to save tokenizer. Default:{TOKENIZER_SAVE_PATH}"
	)
	parser.add_argument(
	    "--resume_from_checkpoint",
	    type=str,
	    default=None,
	    help="Path to a saved model .pt file to resume training from."
	)

	return parser


def _download_ebooks(ebook_links: list) -> str:
	"""Simple helper function to download ebooks from Project Gutenberg."""
	
	corpus = ""
	for link in ebook_links:
		if link:
			response = requests.get(link)
			# TODO: Add error handling

			text = response.text
			start_idx = text.find("START OF THE PROJECT GUTENBERG EBOOK")
			end_idx = text.find("End of the Project Guetnberg")
			text = text[start_idx:end_idx]
			corpus += text + " "

	return corpus


def main():
	"""
	Sample command: 
	```
	python utils/training.py --seq_len 100 --batch_size 8 --epochs 1 
	```
	"""
	
	# Initialize parser
	parser = _setup_parser()
	args = parser.parse_args()
	args = vars(args)

	# Download the ebook(s) from Project Gutenberg
	ebook_link_1 = args.get("ebook_link_1")
	ebook_link_2 = args.get("ebook_link_2")
	ebook_link_3 = args.get("ebook_link_3")
	ebook_link_4 = args.get("ebook_link_4")
	ebook_link_5 = args.get("ebook_link_5")

	ebook_links = [
		ebook_link_1, 
		ebook_link_2,
		ebook_link_3,
		ebook_link_4,
		ebook_link_5
	]

	corpus = _download_ebooks(ebook_links)
	print("Ebooks downloaded successfully!")

	# Parse and load the corpus into a dataloader
	seq_len = args.get("seq_len")
	chunk_size = args.get("chunk_size")
	batch_size = args.get("batch_size")
	tokenizer_save_path = args.get("tokenizer_save_path")

	encoder = TokenEncoder(corpus, seq_len, chunk_size)
	with open(tokenizer_save_path, "wb") as f:
		pickle.dump(encoder, f)
	print(f"Tokenizer saved to {tokenizer_save_path}.")

	ds = DatasetForCausalLM(encoder.chunks)
	dl = load_data(ds, batch_size)

	# Initialize model
	d_model = args.get("d_model")
	nhead = args.get("nhead")
	dim_feedforward = args.get("dim_feedforward")
	num_encoder_layers = args.get("num_encoder_layers")

	model = CausalTransformer(
        input_size=len(encoder.tokens_array),
        d_model=d_model,
        max_seq_len=seq_len,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_encoder_layers		
	)

	# Load model weights from checkpoint
	resume_from_checkpoint = args.get("resume_from_checkpoint")
	if resume_from_checkpoint:
		model.load_state_dict(torch.load(resume_from_checkpoint, weights_only=True))
		print(f"Model loaded from {resume_from_checkpoint}.")

	# Define optimizer and loss
	lr = args.get("lr")

	optimizer = configure_optimizer(model, lr)
	loss_fn = torch.nn.CrossEntropyLoss()

	# Train model
	epochs = args.get("epochs")
	log_interval = args.get("log_interval")

	train(
		model,
		optimizer,
		loss_fn, 
		dl,
		epochs,
		log_interval
	)

	# Save model
	model_save_path = args.get("model_save_path")
	torch.save(model.state_dict(), model_save_path)
	print(f"Model saved to {model_save_path}.")


if __name__ == "__main__":
	main()