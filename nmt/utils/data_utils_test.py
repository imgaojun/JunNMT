from data_utils import TrainDataSet
import vocab_utils
src_vocab_table = vocab_utils.VocabTable('/home/xiapeng/python/process_data/res/vocab.txt')
tgt_vocab_table = vocab_utils.VocabTable('/home/xiapeng/python/process_data/res/vocab.txt')

dataset = TrainDataSet('/home/xiapeng/python/process_data/res/deve_src_file','/home/xiapeng/python/process_data/res/deve_tgt_file',10,src_vocab_table,tgt_vocab_table)

while True:
    try:
        src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var = dataset.iterator
        print(src_input_var)
    except StopIteration:
        print('end of epoch')
        break
        # dataset.init_iterator()
