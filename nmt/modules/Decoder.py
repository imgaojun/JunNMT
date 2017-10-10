import torch
import torch.nn as nn
from Attention import GlobalAttention
from cuda_functional import SRU, SRUCell


class AttnDecoderGRU(nn.Module):
    def __init__(self, attn_model, embeddings, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(AttnDecoderGRU, self).__init__()

        self.bidirectional_encoder = True

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embeddings = embeddings

        self.gru =  nn.GRU(input_size, hidden_size, num_layers, dropout=self.dropout, bidirectional=False)
        self.attention = GlobalAttention(hidden_size)
    def forward(self, rnn_input, last_hidden, encoder_outputs):
        embeded = self.embeddings(rnn_input)        
        rnn_output , hidden = self.gru(embeded,last_hidden)
        attn_output, align_vectors = self.attention(rnn_output.transpose(0,1), encoder_outputs.transpose(0,1))

        return attn_output, hidden
    
class AttnDecoderSRU(nn.Module):
    def __init__(self, attn_model, embeddings, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(AttnDecoderSRU, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embeddings = embeddings
        self.sru = SRU(input_size, hidden_size,
                        num_layers = num_layers,        # number of stacking RNN layers
                        dropout = dropout,           # dropout applied between RNN layers
                        rnn_dropout = dropout,       # variational dropout applied on linear transformation
                        use_tanh = 1,            # use tanh?
                        use_relu = 0,            # use ReLU?
                        bidirectional = False    # bidirectional RNN ?
                    )
        self.attention = GlobalAttention(hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, rnn_input, last_hidden, encoder_outputs):
        embeded = self.embeddings(rnn_input)
        rnn_output , hidden = self.sru(embeded,last_hidden)

        attn_h, align_vectors = self.attention(rnn_output.transpose(0,1), encoder_outputs.transpose(0,1))
        output = self.linear_out(attn_h)

        return output, hidden
