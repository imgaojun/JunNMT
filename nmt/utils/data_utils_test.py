from data_utils import NMTDataSet


dataset = NMTDataSet('/home/xiapeng/python/process_data/res/deve_src_file','/home/xiapeng/python/process_data/res/deve_tgt_file',10)

while True:
    try:
        dataset.iterator
        print('dataset.iterator')
    except StopIteration:
        print('end of epoch')
        dataset.init_iterator()
