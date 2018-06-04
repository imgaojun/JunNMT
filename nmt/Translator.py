import torch
import nmt
import nmt.IO as IO


class Translator(object):
    """
    Uses a model to translate a batch of sentences.
    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, cuda=False,
                 beam_trace=False, min_length=0):
        self.model = model
        # Set model in eval mode.
        self.model.eval()
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": []}

    def translate_batch(self, batch):
        """
        Translate a batch of sentences.
        Mostly a wrapper around :obj:`Beam`.
        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.

        src = batch.src[0]
        src_lengths = batch.src[1].tolist()

        beam_size = self.beam_size
        batch_size = len(src_lengths)
        vocab = self.fields["tgt"].vocab
        beam = [nmt.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[IO.PAD_WORD],
                                    eos=vocab.stoi[IO.EOS_WORD],
                                    bos=vocab.stoi[IO.BOS_WORD],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return a

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.

        context, enc_states = self.model.encode(src, src_lengths)
        dec_states = self.model.init_decoder_state(enc_states, context)

        # (2) Repeat src objects `beam_size` times.

        context = rvar(context.data)
        src_lengths = torch.LongTensor(src_lengths).cuda()
        if not isinstance(dec_states, tuple): # GRU
            dec_states = dec_states.data.repeat(1, beam_size, 1)
        else: # LSTM
            dec_states = (
                dec_states[0].data.repeat(1, beam_size, 1),
                dec_states[1].data.repeat(1, beam_size, 1),
                )

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))


            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            # inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.model.decode(inp, context, dec_states)
            if not isinstance(dec_states, tuple): # GRU
                dec_states = [ 
                    dec_states
                ]   
            else:
                dec_states = [ 
                    dec_states[0],
                    dec_states[1]
                ]

            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            out = self.model.generator(dec_out).data
            out = unbottle(out)
            # beam x tgt_vocab


            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(
                    out[:, j])
                dec_states = self.beam_update(j, b.get_current_origin(), beam_size, dec_states)

            if not isinstance(dec_states, tuple): # GRU
                dec_states = dec_states[-1]
            else:
                dec_states = ( 
                    dec_states[0],
                    dec_states[1]
                ) 
        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)

        return ret

    def beam_update(self, idx, positions, beam_size,states):
        out = []
        for e in states:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))
            out.append(sent_states)
        return tuple(out)

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": []}
        for b in beam:
            if self.beam_accum:
                self.beam_accum['predicted_ids'].append(torch.stack(b.next_ys[1:]).tolist())
                self.beam_accum['beam_parent_ids'].append(torch.stack(b.prev_ks).tolist())
                self.beam_accum['scores'].append(torch.stack(b.all_scores).tolist())

            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp = b.get_hyp(times, k)
                hyps.append(hyp)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
        return ret