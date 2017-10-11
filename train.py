import nmt.utils.model_utils as model_utils
import nmt.utils.misc_utils as utils
import nmt.utils.vocab_utils as vocab_utils
import nmt.utils.data_utils as data_utils
import argparse
from torch import optim
from nmt.Trainer import Trainer
from nmt.Criteria import Criteria
train_parser = argparse.ArgumentParser()
train_parser.add_argument("--config", type=str, default="./config.yml")
args = train_parser.parse_args()
hparams = utils.load_hparams(args.config)

src_vocab_table = vocab_utils.VocabTable(hparams['src_vocab_file'])
tgt_vocab_table = vocab_utils.VocabTable(hparams['tgt_vocab_file'])

dataset = data_utils.TrainDataSet(hparams['dev_src_file'],hparams['dev_tgt_file'],10,src_vocab_table,tgt_vocab_table)


if __name__ == '__main__':
    train_model = model_utils.create_base_model(hparams,src_vocab_table.vocab_size,tgt_vocab_table.vocab_size)
    train_criteria = Criteria()
    optim = optim.Adam(train_model.parameters(), lr=hparams['learning_rate'])
    trainer = Trainer(train_model,dataset.iterator,None,train_criteria,None,optim)
