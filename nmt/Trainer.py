import time
import nmt.utils.vocab_utils as vocab_utils
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

    def print_out(self, epoch, batch, n_batches, start, summary_writer=None):
        t = self.elapsed_time()

        out_info = ("Epoch %2d, %5d/%5d| acc: %6.2f| ppl: %6.2f| " +
               "%3.0f tgt tok/s| %4.0f s elapsed") %
              (epoch, batch, n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_words / (t + 1e-5),
               time.time() - self.start_time)

        # print(("Epoch %2d, %5d/%5d| acc: %6.2f| ppl: %6.2f| " +
        #        "%3.0f tgt tok/s| %4.0f s elapsed") %
        #       (epoch, batch, n_batches,
        #        self.accuracy(),
        #        self.ppl(),
        #        self.n_words / (t + 1e-5),
        #        time.time() - self.start_time))
        print(out_info)
        if summary_writer is not None:
            summary_writer.add_text('progress',out_info,epoch)
        sys.stdout.flush()

    def log(self, prefix, summary_writer, step, **kwargs):

        for key in kwargs:
            summary_writer.add_scalar(prefix + '/' + key, kwargs[key],step)

class Trainer(object):
    def __init__(self, 
                 hparams,
                 model, 
                 train_dataset,
                 valid_dataset, 
                 train_criterion, 
                 valid_criterion,
                 optim):

        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_criterion = train_criterion
        self.valid_criterion = valid_criterion
        self.optim = optim

        self.USE_CUDA = hparams['USE_CUDA']

        self.out_dir = hparams['out_dir']

        # Set model in training mode.
        self.model.train()       

        self.global_step = 0
        self.step_epoch = 0

    def update(self,
               src_inputs,
               src_lengths,
               tgt_inputs,
               tgt_lengths,
               tgt_outputs):
        self.model.zero_grad()
        batch_size = src_inputs.size(1)
        all_decoder_outputs = self.model(src_inputs,tgt_inputs,src_lengths)

        loss, stats = self.train_criterion.compute_loss(all_decoder_outputs.transpose(0, 1).contiguous(), 
                                                        tgt_outputs.transpose(0, 1).contiguous(), 
                                                        tgt_lengths)
        loss = loss.div(batch_size)
        loss.backward()
        self.optim.step()
        return stats

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        step_batch = 0
        while True:
            try:
                src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var = self.train_dataset.iterator

            except StopIteration:     
                print('end of epoch')  
                break                 

            step_batch += 1
            self.global_step += 1

            stats = self.update(src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var)

            report_stats.update(stats)
            total_stats.update(stats)

            if report_func is not None:
                report_stats = report_func(self.global_step,
                        epoch, step_batch, len(self.train_dataset.train_iter),
                        total_stats.start_time, self.optim.lr, report_stats) 


        return total_stats           

    def validate(self):
        self.model.eval()
        valid_stats = Statistics()

        while True:
            try:
                src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var = self.valid_dataset.iterator

            except StopIteration:     
                print('end of epoch')  
                break                 

            all_decoder_outputs = self.model(src_input_var,tgt_input_var,src_input_lengths)
            stats = self.valid_criterion.compute_valid_loss(all_decoder_outputs.transpose(0, 1).contiguous(), 
                                                     tgt_output_var.transpose(0, 1).contiguous())
            valid_stats.update(stats)        
        # Set model back to training mode.
        self.model.train()
        return valid_stats

    def save_per_epoch(self, epoch):
        f = open(os.path.join(self.out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.model.save_checkpoint(epoch, 
                    os.path.join(self.out_dir,"checkpoint_epoch%d.pkl"%(epoch)))
        
        

    
    def load_checkpoint(self, filenmae):
        self.model.load_checkpoint(filenmae)
        

    def epoch_step(self, ppl, epoch):
        """ Called for each epoch to update learning rate. """
        self.optim.updateLearningRate(ppl, epoch) 
        self.save_per_epoch(epoch)
        self.train_dataset.init_iterator()
        self.valid_dataset.init_iterator()