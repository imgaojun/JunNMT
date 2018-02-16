import torch
import argparse
import codecs
import nmt
import json
from torch import cuda

def translate_sentence(translator, sent, fields, use_cuda):
    src_input_var, src_input_lengths= \
        nmt.data_utils.batch_seq2var([sent],
                                    fields['src'].vocab.stoi,
                                    use_cuda)

    ret = translator.translate_batch(src_input_var,src_input_lengths)
    # all_hyp_inds = ret['predictions'][0]
    
    # all_hyp_words = [nmt.data_utils.indices2words(idxs,fields['tgt'].vocab.itos) for idxs in all_hyp_inds]
    
    
    # sentence_out = ' '.join(all_hyp_words[0])
    # sentence_out = sentence_out.replace(' <unk>','')
    # sentence_out = sentence_out.replace(' </s>','')
    return ret




def translate_file(translator, src_fin, tgt_fout, fields, use_cuda, dump_beam=None):

    def get_sentence(sent_list):
        all_hyp_inds = sent_list#ret['predictions'][0]
        
        all_hyp_words = [nmt.data_utils.indices2words(idxs,fields['tgt'].vocab.itos) for idxs in all_hyp_inds]
        sentence_out = ' '.join(all_hyp_words[0])
        sentence_out = sentence_out.replace(' <unk>','')
        sentence_out = sentence_out.replace(' </s>','')

        return sentence_out        


    print('start translating ...')
    with codecs.open(src_fin, 'r', encoding='utf8', errors='ignore') as src_file,\
            codecs.open(tgt_fout, 'wb', encoding='utf8') as tgt_file:
        src_lines = src_file.readlines()
        process_bar = nmt.misc_utils.ShowProcess(len(src_lines))
        for line in src_lines:
            src_seq = line.strip()
            ret = translate_sentence(translator, src_seq, fields, use_cuda)
            sentence_out = get_sentence(ret['predictions'][0])

            tgt_file.write(sentence_out+'\n')

            process_bar.show_process()
        if dump_beam:
            print('dump beam ....')
            beam_trace = translator.beam_accum
            print(len(beam_trace['predicted_ids']))
            with codecs.open(dump_beam,'w',encoding='utf8') as f:
                f.write(json.dumps(beam_trace))



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-src_in", type=str)
    parser.add_argument("-tgt_out", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-data", type=str)
    parser.add_argument("-dump_beam", default="", type=str)
    parser.add_argument('-gpuid', default=[], nargs='+', type=int)
    args = parser.parse_args()
    opt = torch.load(args.model)['opt']

    use_cuda = False
    if args.gpuid:
        cuda.set_device(args.gpuid[0])
        use_cuda = True
    fields = nmt.IO.load_fields(
                torch.load(args.data + '.vocab.pkl'))

    model = nmt.model_helper.create_base_model(opt,fields)


    print('Loading parameters ...')

    model.load_checkpoint(args.model)
    if use_cuda:
        model = model.cuda()    

    translator = nmt.Translator(model=model, 
                                fields=fields,
                                beam_size=opt.beam_size, 
                                n_best=1,
                                max_length=opt.decode_max_length,
                                global_scorer=None,
                                cuda=use_cuda,
                                beam_trace=True if args.dump_beam else False)

    translate_file(translator, args.src_in, args.tgt_out, fields, use_cuda, args.dump_beam)
    

if __name__ == '__main__':
    main()