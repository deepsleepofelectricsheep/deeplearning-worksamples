"""Generate text using saved custom causal lm"""
import argparse
import pickle

import torch
import numpy as np

from custom_causal_transformer import CausalTransformer, generate
from tokenizers import TokenEncoder


MODEL_CHECKPOINT = "models/sample_model.pt"
TOKENIZER_CHECKPOINT = "tokenizers/token_encoder.pkl"

SEQ_LEN = 100
D_MODEL = 512
NHEAD = 8
DIM_FEEDFORWARD = 2048
NUM_ENCODER_LAYERS = 6

BLOCK_SIZE = 100
MAX_NEW_TOKENS = 500

INPUT_TEXT = "K. cried in the court of law."


def _setup_parser():
	"""Set up Python's ArgumentParser."""
	
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument(
		"--model_checkpoint",
		type=str,
		default=MODEL_CHECKPOINT,
		help=f"Path to model checkpoint. Default: {MODEL_CHECKPOINT}"
	)
	parser.add_argument(
		"--tokenizer_checkpoint",
		type=str,
		default=TOKENIZER_CHECKPOINT,
		help=f"Path to tokenizer checkpoint. Default: {TOKENIZER_CHECKPOINT}"
	)
	parser.add_argument(
		"--block_size",
		type=int,
		default=BLOCK_SIZE,
		help=f"Length of preceeding sequence for next token generation. Default: {BLOCK_SIZE}"
	)
	parser.add_argument(
		"--max_new_tokens",
		type=int,
		default=MAX_NEW_TOKENS,
		help=f"Maximum number of tokens to generate. Default: {MAX_NEW_TOKENS}"
	)
	parser.add_argument(
		"--input_text",
		type=str,
		default=INPUT_TEXT,
		help=f"User input to inspire text generation. Default: {INPUT_TEXT}"
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
		"--seq_len",
		type=int,
		default=SEQ_LEN,
		help=("Maximum sequence length of input to train the causal lm" 
			  f"to predict the next token. Default: {SEQ_LEN}")
	)

	return parser 


def main():
	"""
	Sample command: 
	```
	python utils/generate.py
	```
	"""
	
	# Initialize parser
	parser = _setup_parser()
	args = parser.parse_args()
	args = vars(args)

	# Load tokenizer
	tokenizer_checkpoint = args.get("tokenizer_checkpoint")
	with open(tokenizer_checkpoint, "rb") as f:
		encoder = pickle.load(f)
	print(f"Tokenizer loaded from {tokenizer_checkpoint}")

	# Encode input
	input_text = args.get("input_text")
	encoded_input = torch.tensor(
		np.array([encoder.encode(input_text)]), 
		dtype=torch.long
	)

	# Initialize model
	seq_len = args.get("seq_len")
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
	model_checkpoint = args.get("model_checkpoint")
	model.load_state_dict(torch.load(model_checkpoint, weights_only=True))
	print(f"Model loaded from {model_checkpoint}")

	# Generate output sequence
	block_size = args.get("block_size")
	max_new_tokens = args.get("max_new_tokens")

	encoded_output = generate(
	    model,
	    encoded_input,
	    block_size,
	    max_new_tokens
	)

	# Decode and print output
	output = " ".join(
		[
			encoder.tokens_array[i] 
			for i in encoded_output[0].tolist()
		]
	)
	print(output)


if __name__ == "__main__":
	main()
