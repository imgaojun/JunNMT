from data_utils import NMTDataSet

src_vocab_table = vocab_utils.VocabTable(hparams['src_vocab_file'],hparams['src_vocab_size'])
tgt_vocab_table = vocab_utils.VocabTable(hparams['tgt_vocab_file'],hparams['tgt_vocab_size'])

dataset = NMTDataSet('/home/xiapeng/python/process_data/res/deve_src_file','/home/xiapeng/python/process_data/res/deve_tgt_file',10)

while True:
    try:
        dataset.iterator
        # print('dataset.iterator')
    except StopIteration:
        print('end of epoch')
        dataset.init_iterator()
