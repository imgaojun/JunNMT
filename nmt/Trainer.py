import time
import nmt.utils.misc_utils as utils
import torch
from torch.autograd import Variable
import random
import os
import sys
import math
class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def ppl(self):
        return utils.safe_exp(self.loss / self.n_words)

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()

        out_info = ("Epoch %2d, %5d/%5d| acc: %6.2f| ppl: %6.2f| " + \
               "%3.0f tgt tok/s| %4.0f s elapsed") % \
              (epoch, batch, n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_words / (t + 1e-5),
               time.time() - self.start_time)

        print(out_info)
        sys.stdout.flush()

    def log(self, prefix, summary_writer, step, **kwargs):

        for key in kwargs:
            summary_writer.add_scalar(prefix + '/' + key, kwargs[key],step)

class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, optim, lr_scheduler):

        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.lr_scheduler = lr_scheduler

        # Set model in training mode.
        self.model.train()       

        self.global_step = 0
        self.step_epoch = 0

    def update(self, batch, shard_size):
        self.model.zero_grad()
        src_inputs = batch.src[0]
        src_lengths = batch.src[1].tolist()
        tgt_inputs = batch.tgt
        outputs = self.model(src_inputs,tgt_inputs,src_lengths)
        stats = self.train_loss.sharded_compute_loss(batch, outputs, shard_size)

        self.optim.step()
        return stats

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()
         
        for batch in self.train_iter:
            self.global_step += 1
            step_batch = batch.iterations
            stats = self.update(batch, 32)
            
            report_stats.update(stats)
            total_stats.update(stats)

            if report_func is not None:
                report_stats = report_func(self.global_step,
                        epoch, step_batch, len(self.train_iter),
                        total_stats.start_time, self.optim.lr, report_stats) 


        return total_stats           

    def validate(self):
        self.model.eval()
        valid_stats = Statistics()

        for batch in self.valid_iter:

            src_inputs = batch.src[0]
            src_lengths = batch.src[1].tolist()
            tgt_inputs = batch.tgt


            outputs = self.model(src_inputs,tgt_inputs,src_lengths)

            stats = self.valid_loss.monolithic_compute_loss(batch, outputs)
            valid_stats.update(stats)        
        # Set model back to training mode.
        self.model.train()
        return valid_stats

    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.model.save_checkpoint(epoch, 
                    os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))
        
        

    
    def load_checkpoint(self, filenmae):
        self.model.load_checkpoint(filenmae)
        

    def epoch_step(self, epoch, out_dir):
        """ Called for each epoch to update learning rate. """
        # self.optim.updateLearningRate(ppl, epoch) 
        # self.lr_scheduler.step()
        self.save_per_epoch(epoch, out_dir)
