from data_utils import NMTDataSet
import vocab_utils
src_vocab_table = vocab_utils.VocabTable('/home/xiapeng/python/process_data/res/vocab.txt')
tgt_vocab_table = vocab_utils.VocabTable('/home/xiapeng/python/process_data/res/vocab.txt')

dataset = NMTDataSet('/home/xiapeng/python/process_data/res/deve_src_file','/home/xiapeng/python/process_data/res/deve_tgt_file',10,src_vocab_table,tgt_vocab_table)

while True:
    try:
        dataset.iterator
        # print('dataset.iterator')
    except StopIteration:
        print('end of epoch')
        dataset.init_iterator()
