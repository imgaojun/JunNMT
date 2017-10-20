import codecs
from torch.autograd import Variable
import torch
USE_CUDA = True
PAD_token = '<PAD>'
UNK_token = '<UNK>'
SOS_token = '<S>'
EOS_token = '</S>'
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2
SOS_ID = 3


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_ID for i in range(max_length - len(seq))]
    return seq

class VocabTable(object):
    def __init__(self, vocab_file, vocab_size=None):
        self.index2word = {}
        self.word2index = {}
        
        self.index2word[PAD_ID] = PAD_token
        self.index2word[EOS_ID] = EOS_token
        self.index2word[UNK_ID] = UNK_token
        self.index2word[SOS_ID] = SOS_token

        self.word2index[PAD_token] = PAD_ID        
        self.word2index[EOS_token] = EOS_ID
        self.word2index[UNK_token] = UNK_ID
        self.word2index[SOS_token] = SOS_ID

        with codecs.open(vocab_file, 'r', encoding='utf8', errors='replace') as vocab_f:
            idx = 0 
            for line in vocab_f:
                word = line.strip()
                if word not in self.word2index:
                    self.word2index[word] = idx+4
                    self.index2word[idx+4] = word
                    if vocab_size is not None:
                        if idx >= vocab_size:
                            break                
                    idx += 1

        self.vocab_size = len(self.word2index)
        print("vocab size %d"%(self.vocab_size))

def index2word(index, vocab_table):
    return vocab_table.index2word[index]

def word2index(word, vocab_table):
    return vocab_table.word2index[word]

def seq2index(seq, vocab_table):
    seq_idx = []
    words_in = seq.split(' ')
    for w in words_in:
        if w in vocab_table.word2index:
            seq_idx.append(vocab_table.word2index[w])
        else:
            seq_idx.append(vocab_table.word2index[UNK_token])

    return seq_idx + [EOS_ID]
def get_tgt_input_seq(seq):
    seq = [SOS_ID] + seq
    return seq
def get_tgt_output_seq(seq):
    seq += [EOS_ID]
    return seq

def batch2var(batch_src_seqs,batch_tgt_seqs ,src_vocab_table,tgt_vocab_table, USE_CUDA=True):
    src_seqs = [seq2index(seq, src_vocab_table) for seq in batch_src_seqs]
    tgt_seqs = [seq2index(seq, tgt_vocab_table) for seq in batch_tgt_seqs]
    seq_pairs = sorted(zip(src_seqs, tgt_seqs), key=lambda p: len(p[0]), reverse=True)
    src_seqs, tgt_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    src_input_lengths = [len(s) for s in src_seqs]
    paded_src_inputs = [pad_seq(s, max(src_input_lengths)) for s in src_seqs]
    
    tgt_inputs = [get_tgt_input_seq(s) for s in tgt_seqs]
    tgt_input_lengths = [len(s) for s in tgt_inputs]
    paded_tgt_inputs = [pad_seq(s, max(tgt_input_lengths)) for s in tgt_inputs]

    tgt_outputs = [get_tgt_output_seq(s) for s in tgt_seqs]
    tgt_output_lengths = [len(s) for s in tgt_outputs]
    paded_tgt_outputs = [pad_seq(s, max(tgt_output_lengths)) for s in tgt_outputs]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    src_input_var = Variable(torch.LongTensor(paded_src_inputs)).transpose(0, 1)
    tgt_input_var = Variable(torch.LongTensor(paded_tgt_inputs)).transpose(0, 1)
    tgt_output_var = Variable(torch.LongTensor(paded_tgt_outputs)).transpose(0, 1)

    if USE_CUDA:
        src_input_var = src_input_var.cuda()
        tgt_input_var = tgt_input_var.cuda() 
        tgt_output_var = tgt_output_var.cuda() 

    return src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var



def src_seq2var(src_seq, vocab_table):
    seq_idx = [seq2index(src_seq, vocab_table)]
    src_input_length = [len(seq_idx)]
    src_input_var = Variable(torch.LongTensor(seq_idx)).transpose(0, 1)
    if USE_CUDA:
        src_input_var = src_input_var.cuda() 
    return src_input_var, src_input_length
