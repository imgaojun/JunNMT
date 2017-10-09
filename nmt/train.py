import torch.utils.data as data
import codecs
from NMTModel import NMTModel
import argparse
import sys
sys.path.append('./utils')
import misc_utils as utils
import vocab_utils
sys.path.append('./modules')
from Embedding import Embedding
from Encoder import EncoderRNN,EncoderSRU
from Decoder import AttnDecoderRNN,AttnDecoderSRU




train_parser = argparse.ArgumentParser()
train_parser.add_argument("--config", type=str, default="./config.yml")
args = train_parser.parse_args()
hparams = utils.load_hparams(args.config)


eval_pairs = []

with codecs.open(hparams['dev_src_file'], 'r', encoding='utf8',errors='replace') as src_f:
    with codecs.open(hparams['dev_tgt_file'], 'r', encoding='utf8',errors='replace') as tgt_f:
        for line in src_f:
            src_seq = line.strip()
            tgt_seq = tgt_f.readline().strip()
            eval_pairs.append((src_seq,tgt_seq))

train_dataset = utils.TrainDataSet(hparams['train_src_file'],hparams['train_tgt_file'])

train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=hparams['batch_size'],
                               shuffle=True,
                               num_workers=4)

src_vocab_table = vocab_utils.VocabTable(hparams['src_vocab_file'],hparams['src_vocab_size'])
tgt_vocab_table = vocab_utils.VocabTable(hparams['tgt_vocab_file'],hparams['tgt_vocab_size'])



if hparams['share_embeddings'] == True:
    share_embeddings = Embedding(src_vocab_table.vocab_size,hparams['embedding_size'])
    encoder_embeddings = share_embeddings
    decoder_embeddings = share_embeddings
    
else:
    encoder_embeddings = Embedding(src_vocab_table.vocab_size,hparams['embedding_size'])
    decoder_embeddings = Embedding(tgt_vocab_table.vocab_size,hparams['embedding_size'])

encoder = EncoderGRU(encoder_embeddings,
                            hparams['embedding_size'],
                            hparams['hidden_size'],
                            hparams['num_layers'],
                            hparams['dropout'])
decoder = AttnDecoderGRU(hparams['atten_model'],
                                decoder_embeddings,
                                hparams['embedding_size'],
                                hparams['hidden_size'],
                                tgt_vocab_table.vocab_size,
                                hparams['num_layers'],
                                hparams['dropout'])

# encoder = EncoderSRU(encoder_embeddings,
#                             hparams['embedding_size'],
#                             hparams['hidden_size'],
#                             hparams['num_layers'],
#                             hparams['dropout'])
# decoder = AttnDecoderSRU(hparams['atten_model'],
#                                 decoder_embeddings,
#                                 hparams['embedding_size'],
#                                 hparams['hidden_size'],
#                                 tgt_vocab_table.vocab_size,
#                                 hparams['num_layers'],
#                                 hparams['dropout'])                                
model = NMTModel(hparams,encoder,decoder,src_vocab_table,tgt_vocab_table)

model.trainEpoch(hparams,train_loader,eval_pairs)