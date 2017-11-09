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
    def __init__(self, rnn_type, attn_model, input_size, 
                hidden_size, num_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()        
        # Basic attributes.
        self.rnn_type = rnn_type
        self.attn_model = attn_model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)  
        # self.merge_net = nn.Sequential(
        #                     nn.Linear(hidden_size, hidden_size),
        #                     nn.ReLU()
        #                 )
        if rnn_type == "SRU":
            self.rnn = SRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)              

        if self.attn_model != 'none':
            self.attn = GlobalAttention(hidden_size, attn_model)

    def forward(self, input, context, state):
        emb = input
        rnn_outputs, hidden = self.rnn(emb, state)
        
        if self.attn_model != 'none':
            # Calculate the attention.
            attn_outputs, attn_scores = self.attn(
                rnn_outputs.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                context.transpose(0, 1)                   # (contxt_len, batch, d)
            )

            outputs  = self.dropout(attn_outputs)    # (input_len, batch, d)

        else:
            outputs  = self.dropout(rnn_outputs)


        return outputs , hidden


    def init_decoder_state(self, enc_hidden):
        if isinstance(enc_hidden, tuple):  # GRU
            # h = self.merge_net(enc_hidden)
            h = enc_hidden
        else:  # LSTM
            h = enc_hidden
        return h


class InputFeedDecoder(DecoderBase):
    def __init__(self, rnn_type, attn_model, input_size, 
                hidden_size, num_layers=1, dropout=0.1):
        super(InputFeedDecoder, self).__init__()    
        # Basic attributes.
        self.rnn_type = rnn_type
        self.attn_model = attn_model
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)  

        if rnn_type == "SRU":
            self.rnn = SRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=self._input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)              

        if self.attn_model != 'none':
            self.attn = GlobalAttention(hidden_size, attn_model)

    def forward(self, input, context, state):
        outputs = []
        output = self.init_input_feed(context).squeeze(0)
        emb  = input
        hidden = state

        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)  

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                                rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                                context.transpose(0, 1)                   # (contxt_len, batch, d)
                            )    

            output = self.dropout(attn_output)
            outputs += [output]
        return outputs, hidden

    def init_decoder_state(self, enc_hidden):
        if isinstance(enc_hidden, tuple):  # GRU
            # h = self.merge_net(enc_hidden)
            h = enc_hidden
        else:  # LSTM
            h = enc_hidden
        return h

    def init_input_feed(self, context):
        batch_size = context.size(1)
        hidden_size = self.hidden_size
        h_size = (batch_size, hidden_size)
        return Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.input_size + self.hidden_size