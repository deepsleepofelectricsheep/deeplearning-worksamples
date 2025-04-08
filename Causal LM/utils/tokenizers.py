"""Simple implementation of a tokenizer class for text data processing."""
import re
from collections import Counter, OrderedDict
import numpy as np


class TokenEncoder:
    def __init__(self, corpus: str, seq_len: int, chunk_size: int) -> None:
        self.tokens = self.tokenizer(corpus)
        self.token_set = sorted(set(self.tokens))
        self.token2int = {tok: i for i, tok in enumerate(self.token_set)}
        self.tokens_array = np.array(self.token_set)
        self.tokens_encoded = np.array(
            [self.token2int[tok] for tok in self.tokens]
        )
        self.chunks = np.array([
            self.tokens_encoded[i:i+seq_len+chunk_size] 
            for i in range(len(self.tokens_encoded)-seq_len-chunk_size+1)
        ])
        
    def encode(self, s: str) -> np.ndarray:
        tokens = self.tokenizer(s)
        return np.array([self.token2int[tok] for tok in tokens if tok in self.token2int])

    def decode(self, indices: np.ndarray) -> list:
        return [self.tokens_array[i] for i in indices if i < len(self.tokens_array)]
    
    @staticmethod
    def tokenizer(s: str) -> list:  
        s = re.sub("<[^>]*>", "", s)
        tokenized = s.split()
        return tokenized