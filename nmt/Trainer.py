import time
import nmt.utils.vocab_utils as vocab_utils
import torch
from torch.autograd import Variable
import random

class Trainer(object):
    def __init__(self, model, 
                 train_dataset, 
                 train_criteria, 
                 optim, 
                 src_vocab_table,
                 tgt_vocab_table,
                 out_dir,
                 USE_CUDA=True):

        self.model = model
        self.train_dataset = train_dataset
        self.train_criteria = train_criteria
        self.optim = optim

        self.src_vocab_table = src_vocab_table
        self.tgt_vocab_table = tgt_vocab_table

        self.USE_CUDA = USE_CUDA

        self.out_dir = out_dir     
        # Set model in training mode.
        self.model.train()       

     
    def update(self,
               src_inputs,
               src_lengths,
               tgt_inputs,
               tgt_lengths,
               tgt_outputs):
        self.optim.zero_grad()
        all_decoder_outputs = self.model(src_inputs,tgt_inputs,src_lengths)

        loss = self.train_criteria(all_decoder_outputs.transpose(0, 1).contiguous(), tgt_outputs.transpose(0, 1).contiguous(), tgt_lengths)
        loss.backward()
        self.optim.step()
        return loss.data[0]

    def train(self,
              hparams,
              eval_dataset = None):

        num_train_epochs = hparams['num_train_epochs']
        steps_per_stats = hparams['steps_per_stats']
        steps_per_eval = hparams['steps_per_eval']

        global_step = 0
        step_epoch = 0
        avg_step_time  = 0.0

        last_stats_step = global_step
        last_eval_step = global_step        
        

        step_time, checkpoint_loss = 0.0, 0.0              
        start_train_time = time.time()
      
        print('start training...')
        while step_epoch < num_train_epochs:
            step_epoch += 1

            while True:
                try:
                    src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var = self.train_dataset.iterator

                except StopIteration:     
                    print('end of epoch')  
                    self.train_dataset.init_iterator()
                    break         

                global_step += 1
                start_time = time.time()                
                batch_size = src_input_var.size(1)
                # Run the update function
                step_loss = self.update(src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var)
                    
                # update statistics
                step_time += (time.time() - start_time)

                # Keep track of loss
                checkpoint_loss += (step_loss * batch_size)

                # Once in a while, we print statistics.
                if global_step - last_stats_step >= steps_per_stats:
                    last_stats_step = global_step
                    # Print statistics for the previous epoch.
                    avg_step_time = step_time / steps_per_stats
                    avg_step_loss = checkpoint_loss / steps_per_stats
                    checkpoint_loss = 0.0
                    step_time = 0.0

                    print("  global step %d epoch %d step-time %.5fs step-loss %.3f"%
                            (global_step,
                            step_epoch,
                            avg_step_time,
                            avg_step_loss)
                        )           
                if global_step - last_eval_step >= steps_per_eval:
                    last_eval_step = global_step
                    if eval_dataset is not None:
                        self.eval_randomly(hparams,eval_dataset)
                        self.eval_randomly(hparams,eval_dataset)
                        self.eval_randomly(hparams,eval_dataset)


            print('saving best model ...')
            self.save_per_epoch(step_epoch)

    def infer(self,
               src_input,
               src_length,
               max_length):

        self.model.eval()
        # Run wrods through encoder
        encoder_outputs, encoder_hidden = self.model.encoder(src_input, src_length, None)    


        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([vocab_utils.SOS_ID])) # SOS
        decoder_hidden = encoder_hidden
        
        if self.USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_words = []

        # Run through decoder
        for di in range(max_length):
            decoder_input = torch.unsqueeze(decoder_input,0)
            decoder_output, decoder_hidden = self.model.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Choose top word from output
            topv, topi = decoder_output.data.topk(2)
            ni = topi[0][0]
            ni = ni.cpu().numpy().tolist()[0]
            if  ni == vocab_utils.EOS_ID:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.tgt_vocab_table.index2word[ni])
                
            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))
            if self.USE_CUDA: decoder_input = decoder_input.cuda()


        self.model.train()

        return decoded_words

    def eval_randomly(self,hparams,eval_dataset):
        input_sentence, target_sentence = random.choice(eval_dataset)
        src_inputs = [vocab_utils.seq2index(input_sentence,self.src_vocab_table)]
        src_input_lengths = [len(src_inputs)]
        
        src_input_var = Variable(torch.LongTensor(src_inputs)).transpose(0, 1)
        if self.USE_CUDA:
            src_input_var = src_input_var.cuda()

        output_words = self.infer(src_input_var,src_input_lengths,hparams['decode_max_length'])
        output_sentence = ' '.join(output_words)
        print('> src: ', input_sentence)
        if target_sentence is not None:
            print('= tgt: ', target_sentence)
        print('< out: ', output_sentence)     


    def save_per_epoch(self, epoch):
        save_dir = os.path.join(self.out_dir,"epoch%d/"%(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir,'model.pkl'))
