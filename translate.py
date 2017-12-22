import torch
import argparse
import codecs
import nmt
from torch import cuda

# parser = argparse.ArgumentParser()
# parser.add_argument("-config", type=str, default="./config.yml")
# parser.add_argument("-src_in", type=str)
# parser.add_argument("-tgt_out", type=str)
# parser.add_argument("-model", type=str)
# parser.add_argument("-data", type=str)
# parser.add_argument('-gpuid', default=[], nargs='+', type=int)
# args = parser.parse_args()
# opt = nmt.misc_utils.load_hparams(args.config)

# use_cuda = False
# if args.gpuid:
#     cuda.set_device(args.gpuid[0])
#     use_cuda = True


# fields = nmt.IO.load_fields(
#             torch.load(args.data + '.vocab.pkl'))

# model = nmt.model_helper.create_base_model(opt,len(fields['src'].vocab), 
#                                            len(fields['tgt'].vocab), 
#                                            fields['tgt'].vocab.stoi[nmt.IO.PAD_WORD])


# print('Loading parameters ...')

# model.load_checkpoint(args.model)

# if args.gpuid:
#     model = model.cuda()

# translator = nmt.Translator(model, 
#                         fields['tgt'].vocab,
#                         opt.beam_size, 
#                         opt.decode_max_length,
#                         opt.replace_unk)



def translate_sentence(translator, sent, fields, use_cuda):
    src_input_var, src_input_lengths= \
        nmt.data_utils.batch_seq2var([sent],
                                    fields['src'].vocab.stoi,
                                    use_cuda)

    hypotheses, scores = translator.translate(src_input_var,src_input_lengths,use_cuda)
    all_hyp_inds = [[x[0] for x in hyp] for hyp in hypotheses]
    
    all_hyp_words = [nmt.data_utils.indices2words(idxs,fields['tgt'].vocab.itos) for idxs in all_hyp_inds]
    
    
    sentence_out = ' '.join(all_hyp_words[0])
    sentence_out = sentence_out.replace(' <unk>','')
    sentence_out = sentence_out.replace(' </s>','')
    return sentence_out

def translate_file(translator, src_fin, tgt_fout, fields, use_cuda):

    print('start translating ...')
    with codecs.open(src_fin, 'r', encoding='utf8', errors='ignore') as src_file,\
            codecs.open(tgt_fout, 'wb', encoding='utf8') as tgt_file:
        src_lines = src_file.readlines()
        process_bar = nmt.misc_utils.ShowProcess(len(src_lines))
        for line in src_lines:
            src_seq = line.strip()
            sentence_out = translate_sentence(translator, src_seq, fields, use_cuda)

            # src_input_var, src_input_lengths= \
            #     nmt.data_utils.batch_seq2var([src_seq],
            #                                 fields['src'].vocab.stoi,
            #                                 use_cuda)

            # hypotheses, scores = translator.translate(src_input_var,src_input_lengths,use_cuda)
            # all_hyp_inds = [[x[0] for x in hyp] for hyp in hypotheses]
            
            # all_hyp_words = [nmt.data_utils.indices2words(idxs,fields['tgt'].vocab.itos) for idxs in all_hyp_inds]
            
            
            # sentence_out = ' '.join(all_hyp_words[0])
            # sentence_out = sentence_out.replace(' <unk>','')
            # sentence_out = sentence_out.replace(' </s>','')

            tgt_file.write(sentence_out+'\n')

            process_bar.show_process()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="./config.yml")
    parser.add_argument("-src_in", type=str)
    parser.add_argument("-tgt_out", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-data", type=str)
    parser.add_argument('-gpuid', default=[], nargs='+', type=int)
    args = parser.parse_args()
    opt = nmt.misc_utils.load_hparams(args.config)

    use_cuda = False
    if args.gpuid:
        cuda.set_device(args.gpuid[0])
        use_cuda = True
    fields = nmt.IO.load_fields(
                torch.load(args.data + '.vocab.pkl'))

    model = nmt.model_helper.create_base_model(opt,len(fields['src'].vocab), 
                                            len(fields['tgt'].vocab), 
                                            fields['tgt'].vocab.stoi[nmt.IO.PAD_WORD])


    print('Loading parameters ...')

    model.load_checkpoint(args.model)
    if use_cuda:
        model = model.cuda()    

    translator = nmt.Translator(model, 
                        fields['tgt'].vocab,
                        opt.beam_size, 
                        opt.decode_max_length,
                        opt.replace_unk)

    translate_file(translator, args.src_in, args.tgt_out, fields, use_cuda)
    

if __name__ == '__main__':
    main()