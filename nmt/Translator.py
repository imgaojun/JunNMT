import nmt.utils.vocab_utils as vocab_utils
import torch
from torch.autograd import Variable
from nmt.modules.Beam import Beam
import torch.nn.functional as F
class Translator(object):
    def __init__(self, model, tgt_vocab_table, beam_size, max_length, replace_unk=True, USE_CUDA=True):
        self.model = model
        self.tgt_vocab_table = tgt_vocab_table
        self.beam_size = beam_size
        self.USE_CUDA = USE_CUDA
        self.replace_unk = replace_unk
        self.max_length = max_length
        self.model.eval()
    def translate(self, src_input, src_input_lengths=None):
        beam_size = self.beam_size
        encoder_outputs, encoder_hidden = self.model.encode(src_input, src_input_lengths, None)
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


        dec_out = dec_states[0][-1].squeeze(0)


        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(self.max_length):

            input = torch.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)

            # print('input')
            # print(input)
            # print('dec_states')
            # print(dec_states[0])
            decoder_output, decoder_hidden = self.model.decode(
                Variable(input), 
                context, 
                dec_states[0]
            )

            dec_states = [
                decoder_hidden
            ]
            dec_out = decoder_output.squeeze(0)

            out = F.softmax(self.model.generator(dec_out)).unsqueeze(0)


            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break         

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}                   


            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.model.decoder.hidden_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))   


            dec_states = (
                update_active(dec_states[0]),
            )
            dec_out = update_active(dec_out)
            context = update_active(context)

            remaining_sents = len(active)                         

        #  (4) package everything up

        allHyp, allScores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()

            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        return allHyp, allScores



        # # Create starting vectors for decoder
        # decoder_input = Variable(torch.LongTensor([vocab_utils.SOS_ID]), volatile=True) # SOS
        # if self.USE_CUDA:
        #     decoder_input = decoder_input.cuda()
        
        # if self.beam_size == 1:
        #     decoded_words = self._greedy_decode(decoder_init_hidden,decoder_input, encoder_outputs)
        # else:
        #     pass

        # return decoded_words


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

