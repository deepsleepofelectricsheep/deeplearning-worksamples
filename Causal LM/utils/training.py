"""Download ebook(s) from project gutenberg and train causal lm."""
import argparse

from utils.tokenizers import TokenEncoder
from utils.dataset_modules import DatasetForCausalLM, load_data
from utils.custom_causal_transformer import CausalTransformer, train, generate

import requests


EBOOK_LINKS = [
	"https://www.gutenberg.org/cache/epub/7849/pg7849.txt", 
	"https://www.gutenberg.org/cache/epub/5200/pg5200.txt"
]

SEQ_LEN = 100
CHUNK_SIZE = 1
BATCH_SIZE = 8

D_MODEL = 512
NHEAD = 8
DIM_FEEDFORWARD = 2048
NUM_ENCODER_LAYERS = 6

EPOCHS = 5,
LOG_INTERVAL = 100


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
		help=("Maximum sequence length of input to train the causal lm 
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
	python utils/training.py 
	```
	"""
	
	parser = _setup_parser()
	args = parser.parse_args()

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

	# Parse and load the corpus into a dataloader
	seq_len = args.get("seq_len")
	chunk_size = args.get("chunk_size")
	batch_size = args.get("batch_size")

	encoder = TokenEncoder(corpus, seq_len, chunk_size)
	ds = DatasetForCausalLM(encoder.chunks)
	dl = load_data(ds, batch_size)