import torch
import torch.nn as nn

class NMTModel(nn.Module):
    def __init__(self, embedding_encoder, embedding_decoder, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.embedding_encoder = embedding_encoder
        self.embedding_decoder = embedding_decoder
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.init_weights()

    def forward(self, src_inputs, tgt_inputs, src_lengths):

        # Run wrods through encoder

        encoder_outputs, encoder_hidden = self.encode(src_inputs, src_lengths, None)


        decoder_init_hidden = encoder_hidden
            
        decoder_outputs , decoder_hiddens = self.decode(
                tgt_inputs, encoder_outputs, decoder_init_hidden
            )        

        
        return decoder_outputs


    def encode(self, input, lengths=None, hidden=None):
        emb = self.embedding_encoder(input)
        encoder_outputs, encoder_hidden = self.encoder(emb, lengths, None)

        return encoder_outputs, encoder_hidden

    def decode(self, input, context, state):
        emb = self.embedding_decoder(input)
        decoder_outputs , decoder_hiddens = self.decoder(
                emb, context, state
            )     

        outputs = self.generator(decoder_outputs)
        return outputs, decoder_hiddens
    
    def save_checkpoint(self, filename):
        torch.save({'encoder_dict': self.encoder.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'embedding_encoder_dict': self.embedding_encoder.state_dict(),
                    'embedding_decoder_dict': self.embedding_decoder.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    },
                   filename)

    def load_checkpoint(self, filename):   
        cpnt = torch.load(filename)
        self.embedding_encoder.load_state_dict(cpnt['embedding_encoder_dict'])
        self.embedding_decoder.load_state_dict(cpnt['embedding_decoder_dict'])
        self.encoder.load_state_dict(cpnt['encoder_dict'])
        self.decoder.load_state_dict(cpnt['decoder_dict'])
        self.generator.load_state_dict(cpnt['generator_dict'])
    def init_weights(self):
        """Initialize weights."""
        # initrange = 0.1
        # self.embedding_encoder.weight.data.uniform_(-initrange, initrange)
        # self.embedding_decoder.weight.data.uniform_(-initrange, initrange)
        pass