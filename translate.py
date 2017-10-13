from nmt.Translator import Translator
import nmt.utils.vocab_utils as vocab_utils
import nmt.model_helper as model_helper
import nmt.utils.misc_utils as utils
from nmt.utils.data_utils import InferDataSet
import argparse
import codecs
infer_parser = argparse.ArgumentParser()
infer_parser.add_argument("--config", type=str, default="./config.yml")
infer_parser.add_argument("--src_in", type=str)
infer_parser.add_argument("--tgt_out", type=str)
infer_parser.add_argument("--model", type=str)

args = infer_parser.parse_args()
hparams = utils.load_hparams(args.config)

src_vocab_table = vocab_utils.VocabTable(hparams['src_vocab_file'])
tgt_vocab_table = vocab_utils.VocabTable(hparams['tgt_vocab_file'])

infer_dataset = InferDataSet(hparams['train_src_file'],
                            hparams['train_tgt_file'],
                            hparams['batch_size'],
                            src_vocab_table)


model = model_helper.create_base_model(hparams, 
                                       src_vocab_table.vocab_size,
                                       tgt_vocab_table.vocab_size)


translator = Translator(model, 
                        tgt_vocab_table, 
                        hparams['beam_size'], 
                        hparams['decode_max_length'])

with codecs.open(args.src_in, 'r', encoding='utf8') as src_file:
    with codecs.open(args.tgt_out, 'r', encoding='utf8') as tgt_file:
        for line in src_file:
            src_seq = line.strip()
            src_input_var, src_input_length= \
                vocab_utils.src_seq2var(src_seq ,src_vocab_table)

            decoded_words = translator.decode(src_input_var, src_input_length)

            sentence_out = ' '.join(decoded_words)
            tgt_file.write(sentence_out+'\n')


