import nmt.utils.misc_utils as utils
import argparse
from tensorboardX import SummaryWriter
import codecs
import os
import shutil
import re
import torch
import torch.nn as nn
from torch import cuda
import nmt
from translate import translate_file
import random
parser = argparse.ArgumentParser()
parser.add_argument("-config", type=str)
parser.add_argument("-nmt_dir", type=str)
parser.add_argument("-data", type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)

args = parser.parse_args()
opt = utils.load_hparams(args.config)

summery_writer = SummaryWriter(opt.log_dir)

use_cuda = False
if args.gpuid:
    cuda.set_device(args.gpuid[0])
    use_cuda = True
    
if opt.random_seed > 0:
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
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
    if batch % opt.steps_per_stats == -1 % opt.steps_per_stats:
        report_stats.print_out(epoch, batch+1, num_batches, start_time)
        report_stats.log("progress", summery_writer, global_step, learning_rate=lr, 
                                                                  ppl=report_stats.ppl(),
                                                                  accuracy=report_stats.accuracy())
        report_stats = nmt.Statistics()

    return report_stats




def make_train_data_iter(train_data, opt):
    """
    This returns user-defined train data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    return nmt.IO.OrderedIterator(
                dataset=train_data, batch_size=opt.batch_size,
                device=args.gpuid[0] if args.gpuid else -1,                
                repeat=False)


def make_valid_data_iter(valid_data, opt):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    return nmt.IO.OrderedIterator(
                dataset=valid_data, batch_size=opt.batch_size,
                device=args.gpuid[0] if args.gpuid else -1,                                
                train=False, sort=False)

def load_fields(train, valid):
    fields = nmt.IO.load_fields(
                torch.load(args.data + '.vocab.pkl'))
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))

    return fields


def build_or_load_model(model_opt, fields):
    # model = build_model(model_opt, fields)
    model = nmt.model_helper.create_base_model(model_opt, 
                                        len(fields['src'].vocab), 
                                        len(fields['tgt'].vocab), 
                                        fields['tgt'].vocab.stoi[nmt.IO.PAD_WORD])
    latest_ckpt = nmt.misc_utils.latest_checkpoint(model_opt.out_dir)
    start_epoch_at = 0
    if model_opt.start_epoch_at is not None:
        ckpt = 'checkpoint_epoch%d.pkl'%(model_opt.start_epoch_at)
        ckpt = os.path.join(model_opt.out_dir,ckpt)
    else:
        ckpt = latest_ckpt
    # latest_ckpt = nmt.misc_utils.latest_checkpoint(model_dir)
    if ckpt:
        print('Loding model from %s...'%(ckpt))
        start_epoch_at = model.load_checkpoint(ckpt)
    else:
        print('Building model...')
    print(model)
        
    return model, start_epoch_at


def build_optim(model, optim_opt):
    optim = nmt.Optim(optim_opt.optim_method, 
                  optim_opt.learning_rate,
                  optim_opt.max_grad_norm,
                  optim_opt.learning_rate_decay,
                  optim_opt.weight_decay,
                  optim_opt.start_decay_at)
                  
    optim.set_parameters(model.parameters())
    return optim

def build_lr_scheduler(optimizer):
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    lambda1 = lambda epoch: epoch // 30
    lambda2 = lambda epoch: 0.95 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, 
                                                  lr_lambda=[lambda2])
    return scheduler    

def check_save_model_path(opt):
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
        print('saving config file to %s ...'%(opt.out_dir))
        # save config.yml
        shutil.copy(args.config, os.path.join(opt.out_dir,'config.yml'))


def test_bleu(model, fields, epoch):
    translator = nmt.Translator(model, 
                            fields['tgt'].vocab,
                            opt.beam_size, 
                            opt.decode_max_length,
                            opt.replace_unk)
    src_fin = opt.multi_bleu_src
    tgt_fout = os.path.join(opt.out_dir,'translate.epoch%d'%(epoch))
    translate_file(translator, src_fin, tgt_fout)   

    output = os.popen('perl %s/tools/multi-bleu.pl %s < %s'%(args.nmt_dir,
                                                             ' '.join(opt.multi_bleu_refs),
                                                             tgt_fout)
                                                             )
    output = output.read()
    # Get bleu value
    bleu_val = re.findall('BLEU = (.*?),',output,re.S)[0]
    bleu_val = float(bleu_val)
    return bleu_val



def train_model(model, train_data, valid_data, fields, optim, lr_scheduler, start_epoch_at):

    train_iter = make_train_data_iter(train_data, opt)
    valid_iter = make_valid_data_iter(valid_data, opt)

    train_loss = nmt.NMTLossCompute(model.generator,fields['tgt'].vocab)
    valid_loss = nmt.NMTLossCompute(model.generator,fields['tgt'].vocab) 

    if use_cuda:
        train_loss = train_loss.cuda()
        valid_loss = valid_loss.cuda()    

    trainer = nmt.Trainer(model,
                        train_iter,
                        valid_iter,
                        train_loss,
                        valid_loss,
                        optim,
                        lr_scheduler)

    num_train_epochs = opt.num_train_epochs
    print('start training...')
    for step_epoch in  range(start_epoch_at+1, num_train_epochs):
        trainer.lr_scheduler.step()
        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(step_epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())

        

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        trainer.epoch_step(valid_stats.ppl(), step_epoch, out_dir=opt.out_dir)
        if opt.test_bleu:
            valid_bleu = test_bleu(model.eval(), fields, step_epoch)
            model.train()

        train_stats.log("train", summery_writer, step_epoch, 
                        ppl=train_stats.ppl(),
                        learning_rate=optim.lr, 
                        accuracy=train_stats.accuracy())
        valid_stats.log("valid", summery_writer, step_epoch, 
                        ppl=valid_stats.ppl(),
                        learning_rate=optim.lr, 
                        bleu=valid_bleu if opt.test_bleu else 0.0,
                        accuracy=valid_stats.accuracy())

        
def main():

    # Load train and validate data.
    print("Loading train and validate data from '%s'" % args.data)
    train = torch.load(args.data + '.train.pkl')
    valid = torch.load(args.data + '.valid.pkl')
    
    # Load fields generated from preprocess phase.
    fields = load_fields(train, valid)

    # Build model.
    model, start_epoch_at = build_or_load_model(opt, fields)
    check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt)
    lr_scheduler = build_lr_scheduler(optim.optimizer)

    if use_cuda:
        model = model.cuda()

    # Do training.
    
    train_model(model, train, valid, fields, optim, lr_scheduler, start_epoch_at)

if __name__ == '__main__':
    main()