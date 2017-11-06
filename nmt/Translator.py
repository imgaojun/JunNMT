import nmt.utils.vocab_utils as vocab_utils
import torch
from torch.autograd import Variable
class Translator(object):
    def __init__(self, model, tgt_vocab_table, beam_size, max_length, replace_unk=True, USE_CUDA=True):
        self.model = model
        self.tgt_vocab_table = tgt_vocab_table
        self.beam_size = beam_size
        self.USE_CUDA = USE_CUDA
        self.replace_unk = replace_unk
        self.max_length = max_length
        self.model.eval()
    def decode(self, src_input, src_input_length=None):
        
        encoder_outputs, encoder_hidden = self.model.encode(src_input, src_input_length, None)
        decoder_init_hidden = self.model.decoder.init_decoder_state(encoder_hidden)

        context_h = encoder_outputs
        batch_size = context_h.size(1)

        # Expand tensors for each beam.
        context = Variable(context_h.data.repeat(1, beam_size, 1))

        dec_states = [
            Variable(decoder_init_hidden.data.repeat(1, beam_size, 1)),
        ]

        beam = [
            Beam(beam_size, cuda=True)
            for k in range(batch_size)
        ]                

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(self.max_length):

            input = torch.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)


        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([vocab_utils.SOS_ID]), volatile=True) # SOS
        if self.USE_CUDA:
            decoder_input = decoder_input.cuda()
        
        if self.beam_size == 1:
            decoded_words = self._greedy_decode(decoder_init_hidden,decoder_input, encoder_outputs)
        else:
            pass

        return decoded_words


    def _greedy_decode(self, decoder_init_hidden, decoder_input, encoder_outputs):
        # Store output words and attention states
        decoded_words = []            
        # Run through decoder
        decoder_hidden = decoder_init_hidden
        for di in range(self.max_length):
            decoder_input = torch.unsqueeze(decoder_input,0)
            decoder_output, decoder_hidden = self.model.decode(
                decoder_input, encoder_outputs, decoder_hidden 
            )

            # Choose top word from output
            topv, topi = decoder_output.data.topk(2)
            topi = topi[0][0].cpu().numpy().tolist()
            ni = topi[0]
  
            if ni == vocab_utils.UNK_ID and self.replace_unk:
                ni == topi[1]
            if ni == vocab_utils.EOS_ID:
                break
            else:

                decoded_words.append(self.tgt_vocab_table.index2word[ni])
                
            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))
            if self.USE_CUDA: decoder_input = decoder_input.cuda()

        return decoded_words
        
        

    def decode_batch(self):
        beam_size = self.beam_size

        pass

