import itchat
import torch
import argparse
import codecs
import nmt
from torch import cuda

parser = argparse.ArgumentParser()
parser.add_argument("-config", type=str, default="./config.yml")
parser.add_argument("-model", type=str)
parser.add_argument("-data", type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)

args = parser.parse_args()
opt = nmt.misc_utils.load_hparams(args.config)

if args.gpuid:
    cuda.set_device(args.gpuid[0])

fields = nmt.IO.load_fields(
            torch.load(args.data + '.vocab.pkl'))

model = nmt.model_helper.create_base_model(opt,len(fields['src'].vocab), 
                                           len(fields['tgt'].vocab), 
                                           fields['tgt'].vocab.stoi[nmt.IO.PAD_WORD])


print('Loading parameters ...')

model.load_checkpoint(args.model)

if opt.use_cuda:
    model = model.cuda()

translator = nmt.Translator(model, 
                        fields['tgt'].vocab,
                        opt.beam_size, 
                        opt.decode_max_length,
                        opt.replace_unk)

def translate(src_seq):
    src_seq = src_seq.strip()
    src_input_var, src_input_lengths= \
        nmt.data_utils.batch_seq2var([src_seq],
                                    fields['src'].vocab.stoi,
                                    opt.use_cuda)
    hypotheses, scores = translator.translate(src_input_var,src_input_lengths,opt.use_cuda)
    all_hyp_inds = [[x[0] for x in hyp] for hyp in hypotheses]
    
    all_hyp_words = [nmt.data_utils.indices2words(idxs,fields['tgt'].vocab.itos) for idxs in all_hyp_inds]
    
    
    sentence_out = ' '.join(all_hyp_words[0])
    # sentence_out = sentence_out.replace(' <unk>','')
    # sentence_out = sentence_out.replace(' </s>','')
    return sentence_out


@itchat.msg_register(itchat.content.TEXT, isGroupChat=True)
def text_reply(msg):
    if msg['isAt']:
        # itchat.send(u'@%s\u2005I received: %s' % (msg['ActualNickName'], msg['Content']), msg['FromUserName'])
        test_str = msg['Content']
        filter_idx = test_str.index(u'\u2005')
        test_str = test_str[filter_idx+1:]
        sentence_out = translate(test_str)

        itchat.send(u'@%s\u2005%s' % (msg['ActualNickName'], sentence_out), msg['FromUserName'])        
        




itchat.auto_login(enableCmdQR=2)
itchat.run()