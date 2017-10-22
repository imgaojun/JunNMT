import torch
import torch.nn as nn
from nmt.modules.Attention import GlobalAttention
from nmt.modules.SRU import SRU
import torch.nn.functional as F


class DecoderBase(nn.Module):
    def forward(self, input, context, state):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        raise NotImplementedError        



class AttnDecoderRNN(DecoderBase):
    """ The GlobalAttention-based RNN decoder. """
    def __init__(self, rnn_type, attn_model, embeddings, 
                hidden_size, output_size, 
                num_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()        
        # Basic attributes.
        self.rnn_type = rnn_type
        self.attn_model = attn_model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)  

        if rnn_type == "SRU":
            self.rnn = SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)              

        if self.attn_model != 'none':
            self.attn = GlobalAttention(hidden_size, attn_model)
        self.linear_out = nn.Linear(hidden_size, output_size)   

    def forward(self, input, context, state):
        emb = input
        rnn_outputs, hidden = self.rnn(emb, state)
        
        if self.attn_model != 'none':
            # Calculate the attention.
            attn_outputs, attn_scores = self.attn(
                rnn_outputs.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                context.transpose(0, 1)                   # (contxt_len, batch, d)
            )

            rnn_outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        outputs = self.linear_out(rnn_outputs)

        return outputs, hidden
