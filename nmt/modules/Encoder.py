import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nmt.modules.SRU import SRU



class EncoderLSTM(nn.Module):
    def __init__(self, embeddings, input_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(EncoderLSTM, self).__init__()
        

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embeddings = embeddings 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=self.dropout, bidirectional=bidirectional)
        
    def forward(self, rnn_input, input_lengths, last_hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embeded = self.embeddings(rnn_input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_lengths)
        outputs, (hidden,c_n) = self.lstm(packed, None)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, (hidden,c_n)

class EncoderGRU(nn.Module):
    def __init__(self, embeddings, input_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(EncoderGRU, self).__init__()
        
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embeddings = embeddings 
<<<<<<< HEAD
        self.gru = nn.RNN(input_size, hidden_size, num_layers, dropout=self.dropout, bidirectional=bidirectional)
=======
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=self.dropout, bidirectional=bidirectional)
>>>>>>> a9896cb1a13567c621bc9ad6a705bceca0828d79
        
    def forward(self, rnn_input, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embeded = self.embeddings(rnn_input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_lengths)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
   
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)   
            hidden = hidden[:, :, :self.hidden_size] + hidden[:, : ,self.hidden_size:]     
        return outputs, hidden

class EncoderSRU(nn.Module):
    def __init__(self, embeddings, input_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(EncoderSRU, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        # SRU doesn't support PackedSequence.
        self.rnn = SRU(input_size, hidden_size,
                        num_layers = num_layers,          # number of stacking RNN layers
                        dropout = dropout,           # dropout applied between RNN layers
                        rnn_dropout = dropout,       # variational dropout applied on linear transformation
                        use_tanh = 1,            # use tanh?
                        use_relu = 0,            # use ReLU?
                        bidirectional = bidirectional    # bidirectional RNN ?
                    )

        self.embeddings = embeddings
        
    def forward(self, rnn_input, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embeded = self.embeddings(rnn_input)
        outputs, hidden = self.rnn(embeded)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
            hidden = hidden[:, :, :self.hidden_size] + hidden[:, : ,self.hidden_size:]     
            
        return outputs, hidden