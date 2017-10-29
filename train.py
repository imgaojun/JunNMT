import nmt.model_helper as model_helper
import nmt.utils.misc_utils as utils
import nmt.utils.vocab_utils as vocab_utils
import nmt.utils.data_utils as data_utils
import argparse
from nmt.Trainer import Trainer
from nmt.Loss import NMTLossCompute
from nmt.Optim import Optim
from nmt.Trainer import Statistics
from tensorboardX import SummaryWriter
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

# print vocab info
print("src_vocab_size %d, tgt_vocab_size %d"%(src_vocab_table.vocab_size, \
                                                tgt_vocab_table.vocab_size))


print('Loading training data ...')
print("train_src_file: %s"%(hparams['train_src_file']))
print("train_tgt_file: %s"%(hparams['train_tgt_file']))
dataset = data_utils.TrainDataSet(hparams['train_src_file'],
                                  hparams['train_tgt_file'],
                                  hparams['batch_size'],
                                  src_vocab_table,
                                  tgt_vocab_table,
                                  hparams['src_max_len'],
                                  hparams['tgt_max_len'])



summery_writer = SummaryWriter()


def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % hparams['steps_per_stats'] == -1 % hparams['steps_per_stats']:
        report_stats.print_out(epoch, batch+1, num_batches, start_time)
        report_stats.log("progress", summery_writer, lr)
        report_stats = Statistics()

    return report_stats


def train_model(model, train_criterion, optim):


    
    trainer = Trainer(hparams,
                      model,
                      dataset,
                      train_criterion,
                      optim,
                      src_vocab_table,
                      tgt_vocab_table)

    num_train_epochs = hparams['num_train_epochs']
    for step_epoch in  range(num_train_epochs):
        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(step_epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())

        trainer.epoch_step(train_stats.ppl(),step_epoch)




if __name__ == '__main__':
    model = model_helper.create_base_model(hparams,src_vocab_table.vocab_size,tgt_vocab_table.vocab_size)
    train_criterion = NMTLossCompute(tgt_vocab_table.vocab_size, vocab_utils.PAD_ID)
    if hparams['USE_CUDA']:
        train_model = model.cuda()
        train_criterion = train_criterion.cuda()

    print("Using %s optim_method, learning_rate %f, max_grad_norm %f"%\
            (hparams['optim_method'], hparams['learning_rate'],hparams['max_grad_norm']))
    optim = Optim(hparams['optim_method'], hparams['learning_rate'],hparams['max_grad_norm'])
    optim.set_parameters(model.parameters())

    
    if not os.path.exists(hparams['out_dir']):
        os.makedirs(hparams['out_dir'])
        print('saving config file to %s ...'%(hparams['out_dir']))
        # save config.yml
        shutil.copy(args.config, hparams['out_dir'])

    train_model(model, train_criterion, optim)