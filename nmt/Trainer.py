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
        self.model.zero_grad()
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
      
        pass