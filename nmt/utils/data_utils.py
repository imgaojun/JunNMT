

class TrainDataSet(data.Dataset):
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

class NMTDataLoader(object):
    def __init__(self):
        self.dataset
        self.dataloader
        pass