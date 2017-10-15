import torch
import torch.nn as nn

class NMTModel(nn.Module):
    def __init__(self, rnn_type, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_type = rnn_type
        

    def forward(self, src_inputs, tgt_inputs, src_lengths):

        # Run wrods through encoder
        if self.rnn_type == 'LSTM':
            encoder_outputs, (encoder_hidden, c_n) = self.encoder(src_inputs, src_lengths, None)

        else:
            encoder_outputs, encoder_hidden = self.encoder(src_inputs, src_lengths, None)

        # decoder_init_hidden = encoder_hidden[:self.encoder.num_layers] # Use last (forward) hidden state from encoder
        decoder_init_hidden = encoder_hidden
        if self.rnn_type == 'LSTM':
            all_decoder_outputs , decoder_hiddens, c_n = self.decoder(
                    tgt_inputs, decoder_init_hidden, c_n, encoder_outputs
                )                    
        else:
            
            all_decoder_outputs , decoder_hiddens = self.decoder(
                    tgt_inputs, decoder_init_hidden, encoder_outputs
                )        


        return all_decoder_outputs
