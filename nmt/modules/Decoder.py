import torch
import torch.nn as nn
from nmt.modules.Attention import GlobalAttention
from nmt.modules.SRU import SRU

class AttnDecoderLSTM(nn.Module):
    def __init__(self, attn_model, embeddings, input_size, hidden_size, output_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(AttnDecoderLSTM, self).__init__()

        self.bidirectional = bidirectional

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embeddings = embeddings

        self.lstm =  nn.LSTM(input_size, hidden_size, num_layers, dropout=self.dropout, bidirectional=bidirectional)
        
        if self.bidirectional:
            self.attention = GlobalAttention(hidden_size*2)
            self.linear_out = nn.Linear(hidden_size*2, output_size)
        else:
            self.attention = GlobalAttention(hidden_size)
            self.linear_out = nn.Linear(hidden_size, output_size)    
    def forward(self, rnn_input, h0, c0, encoder_outputs):
        embeded = self.embeddings(rnn_input)        
        rnn_output , (hidden,c_n) = self.lstm(embeded,(h0,c0))
        attn_h, align_vectors = self.attention(rnn_output.transpose(0,1), encoder_outputs.transpose(0,1))
        output = self.linear_out(attn_h)
        return output, hidden, c_n


class AttnDecoderGRU(nn.Module):
    def __init__(self, attn_model, embeddings, input_size, hidden_size, output_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(AttnDecoderGRU, self).__init__()

        self.bidirectional = bidirectional

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embeddings = embeddings

        self.gru =  nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        
        if self.bidirectional:
            self.attention = GlobalAttention(hidden_size*2)
            self.linear_out = nn.Linear(hidden_size*2, output_size)
        else:
            self.attention = GlobalAttention(hidden_size)
            self.linear_out = nn.Linear(hidden_size, output_size)    
    def forward(self, rnn_input, last_hidden, encoder_outputs):
        embeded = self.embeddings(rnn_input)        
        rnn_output , hidden = self.gru(embeded,last_hidden)
        attn_h, align_vectors = self.attention(rnn_output.transpose(0,1).contiguous(), encoder_outputs.transpose(0,1).contiguous())
        output = self.linear_out(attn_h)
        return output, hidden
    
class AttnDecoderSRU(nn.Module):
    def __init__(self, attn_model, embeddings, input_size, hidden_size, output_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(AttnDecoderSRU, self).__init__()
        self.attn_model = attn_model
        self.bidirectional = bidirectional
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
                        bidirectional = bidirectional    # bidirectional RNN ?
                    )
        
        if self.bidirectional:
            self.attention = GlobalAttention(hidden_size*2)
            self.linear_out = nn.Linear(hidden_size*2, output_size)
        else:
            self.attention = GlobalAttention(hidden_size)
            self.linear_out = nn.Linear(hidden_size, output_size)            

    def forward(self, rnn_input, last_hidden, encoder_outputs):
        embeded = self.embeddings(rnn_input)
        rnn_output , hidden = self.sru(embeded,last_hidden)

        attn_h, align_vectors = self.attention(rnn_output.transpose(0,1).contiguous(), encoder_outputs.transpose(0,1))
        output = self.linear_out(attn_h)

        return output, hidden