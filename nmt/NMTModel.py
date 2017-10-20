import torch
import torch.nn as nn

class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        

    def forward(self, src_inputs, tgt_inputs, src_lengths):

        # Run wrods through encoder

        encoder_outputs, encoder_hidden = self.encoder(src_inputs, src_lengths, None)
            
        all_decoder_outputs , decoder_hiddens = self.decoder(
                tgt_inputs, encoder_outputs, encoder_hidden
            )        

        return all_decoder_outputs
