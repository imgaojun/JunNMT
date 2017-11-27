from nmt.modules.Encoder import EncoderRNN
from nmt.modules.Decoder import AttnDecoderRNN,InputFeedDecoder
from nmt.modules.Embedding import Embedding
from nmt.NMTModel import NMTModel
import torch
import torch.nn as nn




def create_emb_for_encoder_and_decoder(src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       padding_idx):

    embedding_encoder = Embedding(src_vocab_size,src_embed_size,padding_idx)
    embedding_decoder = Embedding(tgt_vocab_size,tgt_embed_size,padding_idx)

        
    return embedding_encoder, embedding_decoder


def create_encoder(hparams):
    
    rnn_type = hparams.rnn_type
    input_size = hparams.embedding_size
    hidden_size = hparams.hidden_size
    num_layers = hparams.num_layers
    dropout = hparams.dropout
    bidirectional = hparams.bidirectional

    encoder = EncoderRNN(rnn_type,
                        input_size,
                        hidden_size,
                        num_layers,
                        dropout,
                        bidirectional)

    return encoder

def create_decoder(hparams):

    decoder_type = hparams.decoder_type
    rnn_type = hparams.rnn_type  
    atten_model = hparams.atten_model
    input_size = hparams.embedding_size
    hidden_size = hparams.hidden_size
    num_layers = hparams.num_layers
    dropout = hparams.dropout 

    if decoder_type == 'AttnDecoderRNN':
        decoder = AttnDecoderRNN(rnn_type,
                                atten_model,
                                input_size,
                                hidden_size,
                                num_layers,
                                dropout)
    elif decoder_type == 'InputFeedDecoder':
        decoder = InputFeedDecoder(rnn_type,
                                atten_model,
                                input_size,
                                hidden_size,
                                num_layers,
                                dropout)        
   

    return decoder

def create_generator(input_size, output_size):
    generator = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LogSoftmax())
    return generator

def weights_init(m):
    pass
    # if isinstance(m, nn.LSTM): 
    #     for p in m.parameters():
    #         nn.init.orthogonal(p.data)

def create_base_model(hparams, src_vocab_size, tgt_vocab_size, padding_idx):
    embedding_encoder, embedding_decoder = \
            create_emb_for_encoder_and_decoder(src_vocab_size,
                                                tgt_vocab_size,
                                                hparams.embedding_size,
                                                hparams.embedding_size,
                                                padding_idx)
    encoder = create_encoder(hparams)
    decoder = create_decoder(hparams)
    generator = create_generator(hparams.hidden_size, tgt_vocab_size)
    model = NMTModel(embedding_encoder, 
                     embedding_decoder, 
                     encoder, 
                     decoder, 
                     generator)
    if hparams.param_init != 0.0:
        print('Intializing model parameters.')
        
        
        for p in model.parameters():
            # p.data.uniform_(-hparams.param_init, hparams.param_init)    
            nn.init.xavier_uniform(p.data)

        # model.apply(weights_init)
    return model

