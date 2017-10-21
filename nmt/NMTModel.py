import torch
import torch.nn as nn

class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.init_weights()

    def forward(self, src_inputs, tgt_inputs, src_lengths):

        # Run wrods through encoder

        encoder_outputs, encoder_hidden = self.encoder(src_inputs, src_lengths, None)


        decoder_init_hidden = encoder_hidden
            
        all_decoder_outputs , decoder_hiddens = self.decoder(
                tgt_inputs, encoder_outputs, decoder_init_hidden
            )        


        return all_decoder_outputs
    

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.encoder.embeddings.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.embeddings.embedding.weight.data.uniform_(-initrange, initrange)