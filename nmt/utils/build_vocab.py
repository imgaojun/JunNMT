# encoding=utf8
import json
import codecs
import sys
import argparse
from collections import OrderedDict
import numpy


parser = argparse.ArgumentParser()  
parser.add_argument("--vocab", help="vocablary file", default="./vocab_file")  
parser.add_argument("--info", help="info file", default="./info.txt")  
parser.add_argument("--src_file")
args = parser.parse_args()  


word_freqs = OrderedDict()
vocab = OrderedDict()
source_max_length = 0
target_max_length = 0



print('build vocabulary, processing', args.src_file)
with codecs.open(args.src_file, 'r',encoding="utf-8", errors='replace') as f:
    for line in f:
        if len(line) > source_max_length:
            source_max_length = len(line)
        words_in = line.strip().split(' ')
        for w in words_in:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1





words = word_freqs.keys()
freqs = word_freqs.values()

sorted_idx = numpy.argsort(freqs)
sorted_words = [words[ii] for ii in sorted_idx[::-1]]

word_dict = codecs.open(args.vocab, 'wb', encoding='utf8')

for ii, ww in enumerate(sorted_words):
    if ww not in vocab:
        vocab[ww] = ii
        word_dict.write(ww+'\n')

    else:
        print("the word %s has been in vocab_set",ww)


# with open(args.vocab, 'wb') as f:
#     json.dump(vocab, f, indent=2, ensure_ascii=False)


print('build vocabulary done'  )  

with codecs.open(args.info, 'wb',encoding="utf-8") as f:
    info_list = []
    info_str = "source_max_length : "+ str(source_max_length)+"\n"
    info_list.append(info_str)
    info_str = "target_max_length : "+ str(target_max_length)+"\n"
    info_list.append(info_str) 
    f.writelines(info_list)
word_dict.close()

print("Done!")