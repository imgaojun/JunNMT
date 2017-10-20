import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size, pad_idx=0):
        super(Embedding, self).__init__()
        self.pad_idx = pad_idx
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=pad_idx)
    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        return embedded
