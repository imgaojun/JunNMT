from nmt.Translator import Translator
import nmt.utils.vocab_utils as vocab_utils
import nmt.model_helper as model_helper
import nmt.utils.misc_utils as utils
import torch
import argparse
import codecs
infer_parser = argparse.ArgumentParser()
infer_parser.add_argument("--config", type=str, default="./config.yml")
infer_parser.add_argument("--src_in", type=str)
infer_parser.add_argument("--tgt_out", type=str)
infer_parser.add_argument("--model", type=str)

args = infer_parser.parse_args()
hparams = utils.load_hparams(args.config)

src_vocab_table = vocab_utils.VocabTable(hparams['src_vocab_file'], hparams['src_vocab_size'])
tgt_vocab_table = vocab_utils.VocabTable(hparams['tgt_vocab_file'], hparams['tgt_vocab_size'])



model = model_helper.create_base_model(hparams, 
                                       src_vocab_table.vocab_size,
                                       tgt_vocab_table.vocab_size)


print('Loading parameters ...')

model.load_checkpoint(args.model)

if hparams['USE_CUDA']:
    model = model.cuda()

translator = Translator(model, 
                        tgt_vocab_table, 
                        hparams['beam_size'], 
                        hparams['decode_max_length'],
                        hparams['replace_unk'])



print('start translating ...')
with codecs.open(args.src_in, 'r', encoding='utf8', errors='ignore') as src_file:
    with codecs.open(args.tgt_out, 'wb', encoding='utf8') as tgt_file:
        for line in src_file:
            src_seq = line.strip()
            src_input_var, src_input_lengths= \
                vocab_utils.src_seq2var([src_seq] ,src_vocab_table)
 
            hypotheses, scores = translator.decode(src_input_var,src_input_lengths)
            all_hyp_inds = [[x[0] for x in hyp] for hyp in hypotheses]
            
            all_hyp_words = [vocab_utils.idxs2words(idxs,tgt_vocab_table) for idxs in all_hyp_inds]
            
            
            sentence_out = ' '.join(all_hyp_words[0])
            sentence_out = sentence_out.replace(' <UNK>','')
            sentence_out = sentence_out.replace(' </S>','')

            tgt_file.write(sentence_out+'\n')


