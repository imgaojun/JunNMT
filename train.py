import nmt.model_helper as model_helper
import nmt.utils.misc_utils as utils
import nmt.utils.vocab_utils as vocab_utils
import nmt.utils.data_utils as data_utils
import argparse
from torch import optim
from nmt.Trainer import Trainer
from nmt.Criteria import Criteria
import codecs
import os
import shutil
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

src_vocab_table = vocab_utils.VocabTable(hparams['src_vocab_file'], hparams['src_vocab_size'])
tgt_vocab_table = vocab_utils.VocabTable(hparams['tgt_vocab_file'], hparams['tgt_vocab_size'])

print('Loading training data ...')
dataset = data_utils.TrainDataSet(hparams['train_src_file'],
                                  hparams['train_tgt_file'],
                                  hparams['batch_size'],
                                  src_vocab_table,
                                  tgt_vocab_table)

if not os.path.exists(hparams['out_dir']):
    os.makedirs(hparams['out_dir'])

print('saving config file to %s ...'%(hparams['out_dir']))
# save config.yml
shutil.copy(args.config, hparams['out_dir'])


if __name__ == '__main__':
    train_model = model_helper.create_base_model(hparams,src_vocab_table.vocab_size,tgt_vocab_table.vocab_size)
    train_criteria = Criteria()
    if hparams['USE_CUDA']:
        train_model = train_model.cuda()

    optim = optim.Adam(train_model.parameters(), lr=hparams['learning_rate'])

    trainer = Trainer(hparams,
                      train_model,
                      dataset,
                      train_criteria,
                      optim,
                      src_vocab_table,
                      tgt_vocab_table)
    trainer.train(hparams,eval_pairs)
