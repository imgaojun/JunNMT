import torch
import torch.nn as nn

class NMTModel(nn.Module):
    def __init__(self, enc_embedding, dec_embedding, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src_inputs, tgt_inputs, src_lengths):

        # Run wrods through encoder

        enc_outputs, enc_hidden = self.encode(src_inputs, src_lengths, None)


        dec_init_hidden = self.init_decoder_state(enc_hidden, enc_outputs)
            
        dec_outputs , dec_hiddens, attn = self.decode(
                tgt_inputs, enc_outputs, dec_init_hidden
            )        

        return dec_outputs, attn



    def encode(self, input, lengths=None, hidden=None):
        emb = self.enc_embedding(input)
        enc_outputs, enc_hidden = self.encoder(emb, lengths, None)

        return enc_outputs, enc_hidden

    def init_decoder_state(self, enc_hidden, context):
        return enc_hidden

    def decode(self, input, context, state):
        emb = self.dec_embedding(input)
        dec_outputs , dec_hiddens, attn = self.decoder(
                emb, context, state
            )     

        return dec_outputs, dec_hiddens, attn
    
    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'encoder_dict': self.encoder.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'enc_embedding_dict': self.enc_embedding.state_dict(),
                    'dec_embedding_dict': self.dec_embedding.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):   
        ckpt = torch.load(filename)
        self.enc_embedding.load_state_dict(ckpt['enc_embedding_dict'])
        self.dec_embedding.load_state_dict(ckpt['dec_embedding_dict'])
        self.encoder.load_state_dict(ckpt['encoder_dict'])
        self.decoder.load_state_dict(ckpt['decoder_dict'])
        self.generator.load_state_dict(ckpt['generator_dict'])
        epoch = ckpt['epoch']
        return epoch

class Generator(nn.Module):
    def __init__(self, num_k, input_szie, output_size):
        super(Generator, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_szie, input_szie) 
                                        for i in range(num_k)])
        self.mix_linear = nn.Linear(input_szie, num_k)
        self.out_linear = nn.Linear(input_szie, output_size)
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, input):
        mix_weight = self.mix_linear(input)
        mix_weight = self.softmax(mix_weight)
        hc = []
        for l in self.linears:
            hck = l(input)
            hc.append(hck)
        hc = torch.stack(hc)
        k_logits = self.out_linear(hc)
        mix_weight = mix_weight.unsqueeze(-1)
        k_logits = k_logits.transpose(0,1)
        soft_k_logits = self.softmax(k_logits)
        mix_logits = mix_weight * soft_k_logits
        mix_logits = mix_logits.sum(1)
        print(mix_logits)
        log_mix_logits = self.log_softmax(mix_logits)
        print(log_mix_logits)
        # print(log_mix_logits)
        raise BaseException("break")
        return log_mix_logits