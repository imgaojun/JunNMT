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
    def __init__(self, loss=0, n_words=0):
        self.loss = loss
        self.n_words = n_words
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, loss, n_src_words, n_words):
        self.n_src_words += n_src_words
        self.loss += loss
        self.n_words += n_words

    def ppl(self):
        return utils.safe_exp(self.loss / self.n_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch, n_batches,
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, summary_writer, lr):
        t = self.elapsed_time()
        summary_writer.add_scalar(prefix + "_ppl", self.ppl())
        summary_writer.add_scalar(prefix + "_tgtper",  self.n_words / t)
        summary_writer.add_scalar(prefix + "_lr", lr)


class Trainer(object):
    def __init__(self, 
                 hparams,
                 model, 
                 train_dataset, 
                 train_criterion, 
                 optim, 
                 src_vocab_table,
                 tgt_vocab_table):

        self.model = model
        self.train_dataset = train_dataset
        self.train_criterion = train_criterion
        self.optim = optim

        self.src_vocab_table = src_vocab_table
        self.tgt_vocab_table = tgt_vocab_table

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
        all_decoder_outputs = self.model(src_inputs,tgt_inputs,src_lengths)

        loss = self.train_criterion.compute_loss(all_decoder_outputs.transpose(0, 1).contiguous(), tgt_outputs.transpose(0, 1).contiguous(), tgt_lengths)
        loss.backward()
        self.optim.step()
        return loss.data[0]

    def train(self, step_epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        for step_batch, batch_inputs in enumerate(self.train_dataset.train_iter):

            src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var \
                            = batch_inputs
            print(src_input_var)
            loss = self.update(src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var)

            report_stats.update(loss,sum(src_input_lengths),sum(tgt_input_lengths))
            total_stats.update(loss,sum(src_input_lengths),sum(tgt_input_lengths))

            if report_func is not None:
                report_stats = report_func(
                        step_epoch, step_batch, len(self.train_dataset.train_iter),
                        total_stats.start_time, self.optim.lr, report_stats) 


        return total_stats           

        # num_train_epochs = hparams['num_train_epochs']
        # steps_per_stats = hparams['steps_per_stats']
        # steps_per_eval = hparams['steps_per_eval']

        # global_step = self.global_step
        # step_epoch = self.step_epoch
        # avg_step_time  = 0.0

        # last_stats_step = global_step
        # last_eval_step = global_step        
        

        # step_time, checkpoint_loss = 0.0, 0.0              
        # start_train_time = time.time()
      

        # while step_epoch < num_train_epochs:
        #     step_epoch += 1

        #     while True:
        #         try:
        #             src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var = self.train_dataset.iterator

        #         except StopIteration:     
        #             print('end of epoch')  
        #             self.train_dataset.init_iterator()
        #             break         

        #         global_step += 1
        #         start_time = time.time()                
        #         batch_size = src_input_var.size(1)
        #         # Run the update function
        #         step_loss = self.update(src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var)
                    
        #         # update statistics
        #         step_time += (time.time() - start_time)

        #         # Keep track of loss
        #         checkpoint_loss += (step_loss * batch_size)

        #         # Once in a while, we print statistics.
        #         if global_step - last_stats_step >= steps_per_stats:
        #             last_stats_step = global_step
        #             # Print statistics for the previous epoch.
        #             avg_step_time = step_time / steps_per_stats
        #             avg_step_loss = checkpoint_loss / steps_per_stats
        #             checkpoint_loss = 0.0
        #             step_time = 0.0

        #             print("  global step %d epoch %d step-time %.5fs step-loss %.3f"%
        #                     (global_step,
        #                     step_epoch,
        #                     avg_step_time,
        #                     avg_step_loss)
        #                 )           
        #         if global_step - last_eval_step >= steps_per_eval:
        #             last_eval_step = global_step

        #     print('saving best model ...')
        #     self.save_per_epoch(step_epoch, global_step)



    def save_per_epoch(self, epoch):
        self.model.save_checkpoint(epoch, 
                    os.path.join(self.out_dir,"checkpoint_epoch%d.pkl"%(epoch)))

    def load_checkpoint(self, filenmae):
        self.step_epoch, self.global_step =self.model.load_checkpoint(filenmae)
        

    def epoch_step(self, ppl, epoch):
        """ Called for each epoch to update learning rate. """
        # self.optim.updateLearningRate(ppl, epoch) 
        self.save_per_epoch(epoch)