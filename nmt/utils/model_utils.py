from nmt.modules.Encoder import EncoderGRU
from nmt.modules.Decoder import AttnDecoderGRU
from nmt.modules.Embedding import Embedding
from nmt.model import NMTModel
def create_emb_for_encoder_and_decoder(share_embedding,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size):
    if share_embedding:
        print("# Use the same source embeddings for target")
        embedding = Embedding(src_vocab_size,src_embed_size)
        embedding_encoder = embedding
        embedding_decoder = embedding  
    else:
        embedding_encoder = Embedding(src_vocab_size,src_embed_size)
        embedding_decoder = Embedding(tgt_vocab_size,tgt_embed_size)
    return embedding_encoder, embedding_decoder


def create_encoder(hparams, embedding):
    return EncoderGRU(embedding,
                            hparams['embedding_size'],
                            hparams['hidden_size'],
                            hparams['num_layers'],
                            hparams['dropout'])

def create_decoder(hparams, embedding):
    return AttnDecoderGRU(hparams['atten_model'],
                                embedding,
                                hparams['embedding_size'],
                                hparams['hidden_size'],
                                hparams['num_layers'],
                                hparams['dropout'])

def create_base_model(hparams,src_vocab_size,tgt_vocab_size):
    embedding_encoder, embedding_decoder = create_emb_for_encoder_and_decoder(hparams['share_embedding'],
                                                                              src_vocab_size,
                                                                              hparams['embedding_size'],
                                                                              tgt_vocab_size,
                                                                              hparams['embedding_size'])
    encoder = create_encoder(hparams, embedding_encoder)
    decoder = create_decoder(hparams, embedding_decoder)
    
    model = NMTModel(encoder, decoder)

    return model

