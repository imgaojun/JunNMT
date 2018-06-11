import torch
from torchtext import data, datasets
import torchtext
from collections import Counter, defaultdict
import codecs
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
    fields["src"] = torchtext.data.Field(pad_token=PAD_WORD, include_lengths=True)
    fields["tgt"] = torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD)
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



def merge_vocabs(vocabs, specials, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.
    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=specials,
                                 max_size=vocab_size)

class NMTDataset(torchtext.data.Dataset):

    # @staticmethod
    # def sort_key(ex):
    #     return data.interleave_keys(len(ex.src), len(ex.tgt))

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return -len(ex.src), -len(ex.tgt)
        return -len(ex.src)

    def __init__(self, src_path, tgt_path, fields, **kwargs):

        make_example = torchtext.data.Example.fromlist
        with open(src_path, 'r', encoding="utf8",errors='ignore') as src_f, \
                open(tgt_path, 'r', encoding="utf8",errors='ignore') as tgt_f: 
            examples = []
            for src,tgt in zip(src_f,tgt_f):
                src,tgt = src.strip(),tgt.strip()
                if len(src) == 0 or len(tgt) == 0:
                    print("%s,%s"%(src,tgt))
                    continue
                examples.append(make_example([src,tgt],fields))
        
        super(NMTDataset, self).__init__(examples, fields, **kwargs)    
    



    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
    

class InferDataset(torchtext.data.Dataset):
    
    
    def __init__(self, data_path, fields,  **kwargs):

        make_example = torchtext.data.Example.fromlist
        with open(data_path, 'r', encoding="utf8",errors='ignore') as src_f:
            examples = []
            for src in src_f:
                src = src.strip().split(' ')
                src = ' '.join(src)
                examples.append(make_example([src,],fields))

        super(InferDataset, self).__init__(examples, fields, **kwargs)    
    
    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return -len(ex.src), -len(ex.tgt)
        return -len(ex.src)


    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

# class OrderedIterator(torchtext.data.Iterator):
#     def create_batches(self):
#         if self.train:
#             self.batches = torchtext.data.pool(
#                 self.data(), self.batch_size,
#                 self.sort_key, self.batch_size_fn,
#                 random_shuffler=self.random_shuffler)
#         else:
#             self.batches = []
#             for b in torchtext.data.batch(self.data(), self.batch_size,
#                                           self.batch_size_fn):
#                 self.batches.append(sorted(b, key=self.sort_key))


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))