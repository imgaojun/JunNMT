import nmt.IO
import argparse
fields = nmt.IO.get_fields()
print("Building Training...")
train = nmt.IO.NMTDataset(
    path='/home/gaojun4ever/Documents/Projects/jupyter/train_data.tsv', format='tsv',
    fields=[('src', fields["src"]),
            ('tgt', fields["tgt"])])    

print("Building Vocab...")   
fields["src"].build_vocab(train,max_size=30000)
fields["tgt"].build_vocab(train,max_size=30000)
print("Building Valid...")
valid = nmt.IO.NMTDataset(
    path='/home/gaojun4ever/Documents/Projects/jupyter/valid_data.tsv', format='tsv',
    fields=[('src', fields["src"]),
            ('tgt', fields["tgt"])])
print("Saving train/valid/fields")
torch.save(save_vocab(fields),open('vocab.pt', 'wb'))
train.fields = []
valid.fields = []
torch.save(train, open('train.pt', 'wb'))
torch.save(valid, open('valid.pt', 'wb'))