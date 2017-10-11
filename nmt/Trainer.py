import time

class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_criteria, valid_criteria, optim):

        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_criteria = train_criteria
        self.valid_criteria = valid_criteria
        self.optim = optim

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

        loss = self.train_criteria(all_decoder_outputs, tgt_outputs, tgt_lengths)
        loss.backward()
        self.optim.step()
        return loss.data[0]

    def train(self,
              num_train_epochs,
              steps_per_stats):

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
                    src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var = self.train_iter
                except StopIteration:     
                    print('end of epoch')  
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
