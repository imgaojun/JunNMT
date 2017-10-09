import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        return embedded
