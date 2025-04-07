""".py file containing pytorch models and associated utility functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4


class PositionalEncoding(torch.nn.Module):
    """Classic Attention-is-all-you-need positional encoding.
    
    Borrowed from FSDL's transformer_util.py file. 
    Refer to https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, persistent: bool = False) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = self.make_pe(d_model=d_model, max_len=max_len)  # (max_len, 1, d_model)
        self.register_buffer(
            "pe", pe, persistent=persistent
        )  # not necessary to persist in state_dict, since it can be remade

    @staticmethod
    def make_pe(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (S, B, d_model)
        assert x.shape[2] == self.pe.shape[2]  # type: ignore
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """Generate a triangular (size, size) mask.

    Borrowed from FSDL's transformer_util.py file. 
    Refer to https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs.
	"""
    
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class CausalTransformer(nn.Module):
    """A Transformer based Causal Language Model for Sequence Generation.
    Loosely based on the original transformer architecture.
    
    Implementation inspired by the ResnetTransfromer in the FSDL course.
    (https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs)
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 512,
        max_seq_len: int = 100,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 6
    ) -> None:
        
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=input_size, 
            embedding_dim=d_model,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, 
            max_len=max_seq_len,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False
        )
        self.fc = nn.Linear(d_model, input_size)
        
        self.init_weights() # improves training stability and convergence
        
    def init_weights(self):
        """Embedding and FC weight initialization to improve training 
        stability and faster convergence.
        
        Borrowed from FSDL's ResnetTransformer implementation.
        """
        
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of causal encoder block."""
        
        B, T = x.shape
        x = x.permute(1, 0) # (seq_len, batch_size)
        x = self.embedding(x) # (seq_len, batch_size, encoding_size)
        mask = generate_square_subsequent_mask(T)
        output = self.encoder(src=x, mask=mask)
        output = self.fc(output)
        
        return output


def train(
    model: torch.nn.Module, 
    optimizer: torch.optim.Adam, 
    loss_fn: torch.nn.CrossEntropyLoss, 
    dl: torch.utils.data.DataLoader,
    epochs: int = 5,
    log_interval: int = 100
) -> list:
    """Basic training loop for Causal LM."""

    avg_loss_hist = []

    model = model.to(DEVICE)
    for epoch in range(epochs):
        running_loss = 0
        for i, (inputs, targets) in enumerate(dl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs.permute(1, 2, 0), targets)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % log_interval == 0:
                avg_loss = running_loss / (i + 1)
                avg_loss_hist.append(avg_loss)
                print(f"Epoch {epoch+1} Step {i} | Average loss: {avg_loss:0.4f}")

            time.sleep(0.01)

    return avg_loss_hist

    
@torch.no_grad()
def generate(
    model: torch.nn.Module,
    encoded_input: torch.Tensor,
    block_size: int,
    max_new_tokens: int
) -> torch.tensor:
    """Auto-regressive next token prediction."""

    model = model.to(DEVICE)
    encoded_input = encoded_input.to(DEVICE)
    model.eval()

    for Sy in range(max_new_tokens):
        logits = model(encoded_input[:, -block_size:])
        next_token = torch.argmax(logits[-1], dim=-1, keepdim=True)
        encoded_input = torch.cat([encoded_input, next_token], dim=1)

    return encoded_input


def configure_optimizer(
	model: torch.nn.Module,
	lr: float = LR
) -> torch.optim.optimizer:
	return torch.optim.AdamW(model.parameters(), lr)