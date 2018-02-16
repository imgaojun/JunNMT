import nmt.IO
import argparse
import nmt.utils.misc_utils as utils
import torch
import json
parser = argparse.ArgumentParser()
parser.add_argument('-train_src', type=str)
parser.add_argument('-train_tgt', type=str)

parser.add_argument('-valid_src', type=str)
parser.add_argument('-valid_tgt', type=str)

parser.add_argument('-save_data', type=str)
parser.add_argument('-config', type=str)
args = parser.parse_args()

opt = utils.load_hparams(args.config)

if opt.random_seed > 0:
    torch.manual_seed(opt.random_seed)

fields = nmt.IO.get_fields()
print("Building Training...")
train = nmt.IO.NMTDataset(
    src_path=args.train_src,
    tgt_path=args.train_tgt,   
    fields=[('src', fields["src"]),
            ('tgt', fields["tgt"])])    
print("Building Vocab...")   
nmt.IO.build_vocab(train, opt)
print("Building Valid...")
valid = nmt.IO.NMTDataset(
    src_path=args.valid_src,
    tgt_path=args.valid_tgt,
    fields=[('src', fields["src"]),
            ('tgt', fields["tgt"])])
print("Saving train/valid/fields")
with open(args.save_data+'.tgt_vocab.json','w') as f:
	json.dump(train.fields['tgt'].vocab.itos,f)
torch.save(nmt.IO.save_vocab(fields),open(args.save_data+'.vocab.pkl', 'wb'))
train.fields = []
valid.fields = []
torch.save(train, open(args.save_data+'.train.pkl', 'wb'))
torch.save(valid, open(args.save_data+'.valid.pkl', 'wb'))