import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence



class EncoderGRU(nn.Module):
    def __init__(self, embeddings, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(EncoderGRU, self).__init__()
        

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embeddings = embeddings 
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, rnn_input, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embeded = self.embeddings(rnn_input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden

