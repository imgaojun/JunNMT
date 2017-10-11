import torch.utils.data as data
import codecs
import vocab_utils
class TextDataSet(data.Dataset):
    def __init__(self, src_file, tgt_file):
        self.src_dataset = []
        self.tgt_dataset = []

        with codecs.open(src_file, 'r', encoding='utf8', errors='replace') as src_f:
            with codecs.open(tgt_file, 'r', encoding='utf8', errors='replace') as tgt_f:
                for src_seq in src_f:
                    tgt_seq = tgt_f.readline()
                    self.src_dataset.append(src_seq.strip())
                    self.tgt_dataset.append(tgt_seq.strip())
    def __getitem__(self, index):
        src_seq = self.src_dataset[index]
        tgt_seq = self.tgt_dataset[index]

        return src_seq, tgt_seq

    def __len__(self):
        return len(self.src_dataset)

class TrainDataSet(object):
    def __init__(self, 
                 src_file, 
                 tgt_file, 
                 batch_size,
                 src_vocab_table,
                 tgt_vocab_table):
        self.train_dataset = TextDataSet(src_file, tgt_file)
        self.train_dataloader = data.DataLoader(dataset=self.train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=4)

        self.train_iter = iter(self.train_dataloader)

        self.src_vocab_table = src_vocab_table
        self.tgt_vocab_table = tgt_vocab_table

    
    @property
    def iterator(self):
        src_seqs,tgt_seqs = self.train_iter.next()
        src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var = \
            vocab_utils.batch2var(src_seqs,tgt_seqs,self.src_vocab_table, self.tgt_vocab_table )
        return src_input_var, src_input_lengths, tgt_input_var, tgt_input_lengths, tgt_output_var

    def init_iterator(self):
        self.train_iter = iter(self.train_dataloader)
