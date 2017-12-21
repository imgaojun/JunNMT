from nmt.modules.Encoder import EncoderRNN
from nmt.modules.Decoder import AttnDecoderRNN,InputFeedDecoder,ScheduledDecoder
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


def create_encoder(opt):
    
    rnn_type = opt.rnn_type
    input_size = opt.embedding_size
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    dropout = opt.dropout
    bidirectional = opt.bidirectional

    encoder = EncoderRNN(rnn_type,
                        input_size,
                        hidden_size,
                        num_layers,
                        dropout,
                        bidirectional)

    return encoder

def create_decoder(opt):

    decoder_type = opt.decoder_type
    rnn_type = opt.rnn_type  
    atten_model = opt.atten_model
    input_size = opt.embedding_size
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    dropout = opt.dropout 

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

    elif decoder_type == 'ScheduledDecoder':
        scheduler_type = opt.ratio_scheduler_type
        decoder = ScheduledDecoder(rnn_type,
                                atten_model,
                                scheduler_type,
                                input_size,
                                hidden_size,
                                num_layers,
                                dropout)                                       
   

    return decoder

def create_generator(input_size, output_size):
    generator = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LogSoftmax(dim=-1))
    return generator

def weights_init(m):
    if isinstance(m, nn.Linear): 
        nn.init.xavier_uniform(m.weight.data)

def create_base_model(opt, src_vocab_size, tgt_vocab_size, padding_idx):
    embedding_encoder, embedding_decoder = \
            create_emb_for_encoder_and_decoder(src_vocab_size,
                                                tgt_vocab_size,
                                                opt.embedding_size,
                                                opt.embedding_size,
                                                padding_idx)
    encoder = create_encoder(opt)
    decoder = create_decoder(opt)
    generator = create_generator(opt.hidden_size, tgt_vocab_size)
    model = NMTModel(embedding_encoder, 
                     embedding_decoder, 
                     encoder, 
                     decoder, 
                     generator)
    # if opt.param_init != 0.0:
    #     print('Intializing model parameters.')
        
        
    #     for p in model.parameters():
    #         p.data.uniform_(-opt.param_init, opt.param_init)    
    #         # nn.init.xavier_uniform(p.data)

    #     model.apply(weights_init)
    return model

