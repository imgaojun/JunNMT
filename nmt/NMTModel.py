import torch
from torch import optim
from torch.autograd import Variable
from masked_cross_entropy import masked_cross_entropy
import random
import os
import sys
import time
sys.path.append('./utils')
import vocab_utils


class NMTModel(object):
    def __init__(self,hparams, encoder, decoder,src_vocab_table, tgt_vocab_table):

        self.USE_CUDA = hparams['USE_CUDA']
        self.out_dir = hparams['out_dir']


        self.src_vocab_table = src_vocab_table
        self.tgt_vocab_table = tgt_vocab_table

        self.encoder = encoder
        self.decoder = decoder

        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=hparams['learning_rate'])
        self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=hparams['learning_rate'])


        if self.USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()



    def update(self, src_inputs,
               src_input_lengths,
               tgt_inputs,
               tgt_input_lengths,
               tgt_outputs):

        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()


        # Get batch size
        batch_size = src_inputs.size(1)

        # Run wrods through encoder
        encoder_outputs, encoder_hidden = self.encoder(src_inputs, src_input_lengths, None)
        
        decoder_init_hidden = encoder_hidden[:self.encoder.num_layers] # Use last (forward) hidden state from encoder
        
        all_decoder_outputs , decoder_hiddens = self.decoder(
                tgt_inputs, decoder_init_hidden, encoder_outputs
            )

        # Loss calculation and backpropagation
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
            tgt_outputs.transpose(0, 1).contiguous(), # -> batch x seq
            tgt_input_lengths
        )

        loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()

        return loss.data[0], batch_size, all_decoder_outputs.data

        
        

    def trainEpoch(self, 
                   hparams, 
                   dataloader,
                   eval_pairs):
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
            for src_seqs,tgt_seqs in dataloader:
                global_step += 1

                start_time = time.time()

                src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var = vocab_utils.batch2index(src_seqs,tgt_seqs,self.src_vocab_table,self.tgt_vocab_table)

                # Run the update function
                step_loss, batch_size, all_decoder_outputs= self.update(src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var)


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

                    print("  global step %d epoch %d lr %g step-time %.5fs step-loss %.3f"%
                            (global_step,
                            step_epoch,
                            hparams['learning_rate'],
                            avg_step_time,
                            avg_step_loss)
                        )
                if global_step - last_eval_step >= steps_per_eval:
                    last_eval_step = global_step
                    self.eval_randomly(hparams,eval_pairs)
                    self.eval_randomly(hparams,eval_pairs)
                    self.eval_randomly(hparams,eval_pairs)

            self.save_per_epoch(step_epoch)
            print('save epoch %d model'%(step_epoch))                        






    def infer(self,src_inputs,src_input_lengths,max_length=25):
        self.encoder.eval()
        self.decoder.eval()

        # Get batch size
        batch_size = src_inputs.size(1)

        # Run wrods through encoder
        encoder_outputs, encoder_hidden = self.encoder(src_inputs, src_input_lengths, None)
            
        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([vocab_utils.SOS_ID]), volatile=True) # SOS
        decoder_hidden = encoder_hidden[:self.encoder.num_layers] # Use last (forward) hidden state from encoder
        
        if self.USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_words = []
        
        # Run through decoder
        for di in range(max_length):
            decoder_input = torch.unsqueeze(decoder_input,0)
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
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



        self.encoder.train()
        self.decoder.train()
        return decoded_words

    def eval_randomly(self,hparams,eval_pairs):
        input_sentence, target_sentence = random.choice(eval_pairs)
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

    def run_external_eval(self):
        pass

    def save_per_epoch(self, epoch):
        save_dir = os.path.join(self.out_dir,"epoch%d/"%(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.encoder.state_dict(), os.path.join(save_dir,'encoder.pkl'))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir,'decoder.pkl'))


