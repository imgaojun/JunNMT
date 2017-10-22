from nmt.modules.Encoder import EncoderRNN
from nmt.modules.Decoder import AttnDecoderRNN
from nmt.modules.Embedding import Embedding
from nmt.NMTModel import NMTModel





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


def create_encoder(hparams, embeddings):
    
    rnn_type = hparams['rnn_type']
    hidden_size = hparams['hidden_size']
    num_layers = hparams['num_layers']
    dropout = hparams['dropout']
    bidirectional = hparams['bidirectional']    

    encoder = EncoderRNN(rnn_type,
                        embeddings,
                        hidden_size,
                        num_layers,
                        dropout,
                        bidirectional)

    return encoder

def create_decoder(hparams, embeddings, tgt_vocab_size):

    rnn_type = hparams['rnn_type']    
    atten_model = hparams['atten_model']
    hidden_size = hparams['hidden_size']
    num_layers = hparams['num_layers']
    dropout = hparams['dropout']    

    decoder = AttnDecoderRNN(rnn_type,
                            atten_model,
                            embeddings,
                            hidden_size,
                            tgt_vocab_size,
                            num_layers,
                            dropout)
   

    return decoder

def create_base_model(hparams,src_vocab_size,tgt_vocab_size):
    embedding_encoder, embedding_decoder = create_emb_for_encoder_and_decoder(hparams['share_embedding'],
                                                                              src_vocab_size,
                                                                              tgt_vocab_size,
                                                                              hparams['embedding_size'],
                                                                              hparams['embedding_size'])
    encoder = create_encoder(hparams, hparams['embedding_size'])
    decoder = create_decoder(hparams, hparams['embedding_size'], tgt_vocab_size)
    
    model = NMTModel(embedding_encoder, embedding_decoder, encoder, decoder)

    return model

