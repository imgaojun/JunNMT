import torch
import argparse
import codecs
import nmt
import json
from torch import cuda
import nmt.utils.misc_utils as utils
import progressbar


def indices_lookup(indices,fields):

    words = [fields['tgt'].vocab.itos[i] for i in indices]
    sent = ' '.join(words)
    sent = sent.replace(' </s>','')
    return sent


def batch_indices_lookup(batch_indices,fields):

    batch_sents = []
    for sent_indices in batch_indices:
        sent = indices_lookup(sent_indices,fields)
        batch_sents.append(sent)
    print(batch_sents)
    raise "break"
    return batch_sents



def translate_file(translator, 
                   data_iter, 
                   test_out, fields, 
                   use_cuda, 
                   dump_beam=None):

    print('start translating ...')
    with codecs.open(test_out, 'wb', encoding='utf8') as tgt_file:
        bar = progressbar.ProgressBar()


        for batch in bar(data_iter):
            ret = translator.translate_batch(batch)
            batch_sents = batch_indices_lookup(ret['predictions'][0], fields)
            tgt_file.writelines(batch_sents)

        if dump_beam:
            print('dump beam ....')
            beam_trace = translator.beam_accum
            with codecs.open(dump_beam,'w',encoding='utf8') as f:
                json.dump(beam_trace,f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-test_data", type=str)
    parser.add_argument("-test_out", type=str)
    parser.add_argument("-config", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-data", type=str)
    parser.add_argument("-dump_beam", default="", type=str)
    parser.add_argument('-gpuid', default=[], nargs='+', type=int)
    args = parser.parse_args()
    opt = utils.load_hparams(args.config)





    use_cuda = False
    if args.gpuid:
        cuda.set_device(args.gpuid[0])
        use_cuda = True
    fields = nmt.IO.load_fields(
                torch.load(args.data + '.vocab.pkl'))

    test_dataset = nmt.IO.InferDataset(
        data_path=args.test_data,
        fields=[('src', fields["src"])])

    test_data_iter = nmt.IO.OrderedIterator(
                dataset=test_dataset, device=args.gpuid[0],
                batch_size=1, train=False, sort=False,
                sort_within_batch=True, shuffle=False)


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

    translate_file(translator, test_data_iter, args.test_out, fields, use_cuda, args.dump_beam)
    

if __name__ == '__main__':
    main()