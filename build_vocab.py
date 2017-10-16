import argparse
from collections import OrderedDict
import numpy
import codecs
parser = argparse.ArgumentParser()  
parser.add_argument("--words_out")  
parser.add_argument("--src_in")  
parser.add_argument("--tgt_in")  

args = parser.parse_args()  


word_freqs = OrderedDict()
vocab = OrderedDict()
word_dict = codecs.open(args.words_out, 'wb', encoding='utf8')


print('build vocabulary, processing %s'%(args.src_in))
with codecs.open(args.src_in, 'r',encoding="utf-8", errors='ignore') as f:
    for line in f:
        words_in = line.strip().split(' ')
        for w in words_in:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1

print('build vocabulary, processing %s'%(args.tgt_in))
with codecs.open(args.tgt_in, 'r',encoding="utf-8", errors='ignore') as f:
    for line in f:
        words_in = line.strip().split(' ')
        for w in words_in:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1            


words = word_freqs.keys()
freqs = word_freqs.values()


sorted_idx = numpy.argsort(freqs)
sorted_words = [words[ii] for ii in sorted_idx[::-1]]

ii = 0
for ww in sorted_words:
    if ww not in vocab:
        vocab[ww] = ii
        word_dict.write(ww+'\n')
        ii += 1
word_dict.close()
    