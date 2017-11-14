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
import re
import torch
import torch.nn as nn
train_parser = argparse.ArgumentParser()
train_parser.add_argument("--config", type=str)
train_parser.add_argument("--nmt_dir", type=str)

args = train_parser.parse_args()
hparams = utils.load_hparams(args.config)

src_vocab_table = vocab_utils.VocabTable(hparams['src_vocab_file'], hparams['src_vocab_size'])
tgt_vocab_table = vocab_utils.VocabTable(hparams['tgt_vocab_file'], hparams['tgt_vocab_size'])

# print vocab info
print("src_vocab_size %d, tgt_vocab_size %d"%(src_vocab_table.vocab_size, \
                                                tgt_vocab_table.vocab_size))


print('Loading training data ...')
print("train_src_file: %s"%(hparams['train_src_file']))
print("train_tgt_file: %s"%(hparams['train_tgt_file']))
train_dataset = data_utils.TrainDataSet(hparams['train_src_file'],
                                  hparams['train_tgt_file'],
                                  hparams['batch_size'],
                                  src_vocab_table,
                                  tgt_vocab_table,
                                  hparams['src_max_len'],
                                  hparams['tgt_max_len'])

valid_dataset = data_utils.TrainDataSet(hparams['dev_src_file'],
                                  hparams['dev_tgt_file'][0],
                                  hparams['batch_size'],
                                  src_vocab_table,
                                  tgt_vocab_table,
                                  hparams['src_max_len'],
                                  hparams['tgt_max_len'])

summery_writer = SummaryWriter(hparams['log_dir'])


       
def report_func(global_step, epoch, batch, num_batches,
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
        report_stats.log("progress", summery_writer, global_step, learning_rate=lr)
        report_stats = Statistics()

    return report_stats


def test_bleu():
    # os.system('export CUDA_VISIBLE_DEVICES=0')
    os.system('python3 %s/translate.py \
                --config %s \
                --src_in %s \
                --tgt_out %s \
                --model %s'%(args.nmt_dir,
                             os.path.join(hparams['out_dir'],'config.yml'),
                             hparams['dev_src_file'],
                             os.path.join(hparams['out_dir'],'translate.tmp'),
                             utils.latest_checkpoint(hparams['out_dir']),
                            )
            )
    output = os.popen('perl %s/tools/multi-bleu.pl %s < %s'%(args.nmt_dir,
                                                             ' '.join(hparams['dev_tgt_file']),
                                                             os.path.join(hparams['out_dir'],'translate.tmp')
                                                             ))
    output = output.read()
    # Get bleu value
    bleu_val = re.findall('BLEU = (.*?),',output,re.S)[0]
    bleu_val = float(bleu_val)
    return bleu_val




def train_model(model, train_criterion, optim):


    
    trainer = Trainer(hparams,
                      model,
                      train_dataset,
                      valid_dataset,
                      train_criterion,
                      optim,
                      src_vocab_table,
                      tgt_vocab_table)

    num_train_epochs = hparams['num_train_epochs']
    for step_epoch in  range(num_train_epochs):
        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(step_epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())

        

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        trainer.epoch_step(valid_stats.ppl(),step_epoch)
        
        valid_bleu = test_bleu()
        train_stats.log("train", summery_writer, step_epoch, learning_rate=optim.lr, )
        valid_stats.log("valid", summery_writer, step_epoch, learning_rate=optim.lr, bleu=valid_bleu)

        


if __name__ == '__main__':
    model = model_helper.create_base_model(hparams,src_vocab_table.vocab_size,tgt_vocab_table.vocab_size)
    train_criterion = NMTLossCompute(tgt_vocab_table.vocab_size, vocab_utils.PAD_ID)
    if hparams['USE_CUDA']:
        model = model.cuda()
        train_criterion = train_criterion.cuda()

    # model = torch.nn.DataParallel(model, dim=1)

    print("Using %s optim_method, learning_rate %f, max_grad_norm %f"%\
            (hparams['optim_method'], hparams['learning_rate'],hparams['max_grad_norm']))
    optim = Optim(hparams['optim_method'], 
                  hparams['learning_rate'],
                  hparams['max_grad_norm'],
                  hparams['learning_rate_decay'],
                  hparams['weight_decay'],
                  hparams['start_decay_at'])
                  
    optim.set_parameters(model.parameters())

    
    if not os.path.exists(hparams['out_dir']):
        os.makedirs(hparams['out_dir'])
        print('saving config file to %s ...'%(hparams['out_dir']))
        # save config.yml
        shutil.copy(args.config, os.path.join(hparams['out_dir'],'config.yml'))

    train_model(model, train_criterion, optim)