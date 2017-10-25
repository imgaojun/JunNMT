from nmt.modules.Encoder import EncoderRNN
from nmt.modules.Decoder import AttnDecoderRNN
from nmt.modules.Embedding import Embedding
from nmt.NMTModel import NMTModel
import torch
import torch.nn as nn




def create_emb_for_encoder_and_decoder(share_embedding,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size):

    embedding_encoder = Embedding(src_vocab_size,src_embed_size)
    embedding_decoder = Embedding(tgt_vocab_size,tgt_embed_size)

    if share_embedding:
        print("# Use the same source embeddings for target")
        
    return embedding_encoder, embedding_decoder


def create_encoder(hparams):
    
    rnn_type = hparams['rnn_type']
    input_size = hparams['embedding_size']
    hidden_size = hparams['hidden_size']
    num_layers = hparams['num_layers']
    dropout = hparams['dropout']
    bidirectional = hparams['bidirectional']    

    encoder = EncoderRNN(rnn_type,
                        input_size,
                        hidden_size,
                        num_layers,
                        dropout,
                        bidirectional)

    return encoder

def create_decoder(hparams):

    rnn_type = hparams['rnn_type']    
    atten_model = hparams['atten_model']
    input_size = hparams['embedding_size']
    hidden_size = hparams['hidden_size']
    num_layers = hparams['num_layers']
    dropout = hparams['dropout']    

    decoder = AttnDecoderRNN(rnn_type,
                            atten_model,
                            input_size,
                            hidden_size,
                            num_layers,
                            dropout)
   

    return decoder

def create_generator(input_size, output_size):
    generator = nn.Linear(input_size, output_size)
    return generator


def create_base_model(hparams,src_vocab_size,tgt_vocab_size):
    embedding_encoder, embedding_decoder = \
            create_emb_for_encoder_and_decoder(hparams['share_embedding'],
                                                src_vocab_size,
                                                tgt_vocab_size,
                                                hparams['embedding_size'],
                                                hparams['embedding_size'])
    encoder = create_encoder(hparams)
    decoder = create_decoder(hparams)
    generator = create_generator(hparams['hidden_size'], tgt_vocab_size)
    model = NMTModel(embedding_encoder, 
                     embedding_decoder, 
                     encoder, 
                     decoder, 
                     generator)

    return model

