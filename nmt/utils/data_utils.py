import nmt
from torch.autograd import Variable
import torch
# Pad a with the PAD symbol
def pad_seq(seq, max_length, padding_idx):
    seq += [padding_idx for i in range(max_length - len(seq))]
    return seq
    
def get_src_input_seq(seq):
    seq = seq
    return seq

def seq2indices(seq, word2index, max_len=None):
    seq_idx = []
    words_in = seq.split(' ')
    if max_len is not None:
        words_in = words_in[:max_len]
    for w in words_in:
        if w in word2index:
            seq_idx.append(word2index[w])

    return seq_idx

def batch_seq2var(batch_src_seqs, word2index, USE_CUDA=True):
    src_seqs = [seq2indices(seq, word2index) for seq in batch_src_seqs]
    src_seqs = sorted(src_seqs, key=lambda p: len(p), reverse=True)
    src_inputs = [get_src_input_seq(s) for s in src_seqs]
    print(src_inputs)
    src_input_lengths = [len(s) for s in src_inputs]
    paded_src_inputs = [pad_seq(s, max(src_input_lengths), word2index[nmt.IO.PAD_WORD]) for s in src_seqs]    
    src_input_var = Variable(torch.LongTensor(paded_src_inputs)).transpose(0, 1)
    if USE_CUDA:
        src_input_var = src_input_var.cuda() 
    return src_input_var, src_input_lengths

def indices2words(idxs, index2word):
    words_list = [index2word[idx] for idx in idxs]
    return words_list