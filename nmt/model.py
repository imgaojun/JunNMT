import torch
import torch.nn as nn

class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_inputs, tgt_inputs, src_lengths):
        # Get batch size
        batch_size = src_inputs.size(1)

        # Run wrods through encoder
        encoder_outputs, encoder_hidden = self.encoder(src_inputs, src_lengths, None)

        decoder_init_hidden = encoder_hidden[:self.encoder.num_layers] # Use last (forward) hidden state from encoder
                
        all_decoder_outputs , decoder_hiddens = self.decoder(
                tgt_inputs, decoder_init_hidden, encoder_outputs
            )        

        return all_decoder_outputs