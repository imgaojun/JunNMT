import torch
from torchtext import data, datasets
import torchtext
from collections import Counter, defaultdict
import codecs
import io
PAD_WORD = '<blank>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__

def get_fields():

    fields = {}
    fields["src"] = torchtext.data.Field(pad_token=PAD_WORD,include_lengths=True)
    fields["tgt"] = torchtext.data.Field(init_token=BOS_WORD, pad_token=PAD_WORD)
    fields["tgt_out"] = torchtext.data.Field(eos_token=EOS_WORD,pad_token=PAD_WORD)
    return fields

def load_fields(vocab):
    vocab = dict(vocab)
    fields = get_fields()
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields

def save_vocab(fields):
    vocab = []
    for k, f in fields.items():
        if 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab
def build_vocab(train, opt):
    fields = train.fields
    fields["src"].build_vocab(train, max_size=opt.src_vocab_size)    
    fields["tgt"].build_vocab(train, max_size=opt.tgt_vocab_size)

class NMTDataset(torchtext.data.Dataset):
    
    
    def __init__(self, src_path, tgt_path, fields, **kwargs):

        make_example = torchtext.data.Example.fromlist
        with io.open(src_path, encoding="utf8",errors='replace') as src_f,io.open(tgt_path, encoding="utf8",errors='replace') as tgt_f: 
            examples = [make_example(list(src,tgt,tgt), fields) for src,tgt in zip(src_f,tgt_f)]
        super(NMTDataset, self).__init__(examples, fields, **kwargs)    
    
    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)


    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(NMTDataset, self).__reduce_ex__()
    

class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
