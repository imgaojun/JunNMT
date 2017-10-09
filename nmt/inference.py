import torch.utils.data as data
import codecs
import torch
from NMTModel import NMTModel
import argparse
import sys
import os
sys.path.append('./utils')
import misc_utils as utils
import vocab_utils
sys.path.append('./modules')
from Embedding import Embedding
from Encoder import EncoderRNN
from Decoder import AttnDecoderRNN

infer_parser = argparse.ArgumentParser()
infer_parser.add_argument("--config", type=str, default="./config.yml")
infer_parser.add_argument("--src_in", type=str)
infer_parser.add_argument("--tgt_out", type=str)
args = infer_parser.parse_args()
hparams = utils.load_hparams(args.config)

src_vocab_table = vocab_utils.VocabTable(hparams['src_vocab_file'])
tgt_vocab_table = vocab_utils.VocabTable(hparams['tgt_vocab_file'])


if hparams['share_embeddings'] == True:
    share_embeddings = Embedding(src_vocab_table.vocab_size,hparams['embedding_size'])
    encoder_embeddings = share_embeddings
    decoder_embeddings = share_embeddings
    
else:
    encoder_embeddings = Embedding(src_vocab_table.vocab_size,hparams['embedding_size'])
    decoder_embeddings = Embedding(tgt_vocab_table.vocab_size,hparams['embedding_size'])


print('Creating Seq2Seq Model...')

encoder = EncoderRNN(encoder_embeddings,
                            hparams['embedding_size'],
                            hparams['hidden_size'],
                            hparams['num_layers'],
                            hparams['dropout'])
decoder = AttnDecoderRNN(hparams['atten_model'],
                                decoder_embeddings,
                                hparams['embedding_size'],
                                hparams['hidden_size'],
                                tgt_vocab_table.vocab_size,
                                hparams['num_layers'],
                                hparams['dropout'])

 
print('Loading parameters ...')

encoder.load_state_dict(torch.load(os.path.join(hparams['out_dir'],'epoch1','encoder.pkl')))

decoder.load_state_dict(torch.load(os.path.join(hparams['out_dir'],'epoch1','decoder.pkl')))

print('Seq2Seq Model init...')

model = NMTModel(hparams,encoder,decoder,src_vocab_table,tgt_vocab_table)

print('Processing sequence ... ')

with codecs.open(args.src_in,'r',encoding='utf8',errors='replace') as src_in:
    with codecs.open(args.tgt_out,'wb',encoding='utf8') as tgt_out:
        for line in src_in:
            src_seq = line.strip()
            src_input_var, src_input_length = vocab_utils.src_seq2var(src_seq,src_vocab_table)
            decoded_words = model.infer(src_input_var,src_input_length)
            output_sentence = ' '.join(decoded_words)
            tgt_out.write(output_sentence+'\n')
            # print('> src: ', src_seq)
            # print('< out', output_sentence) 
            



