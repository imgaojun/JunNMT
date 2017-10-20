import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable




class NMTLossCompute(nn.Module):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, tgt_vocab_size, padding_idx):
        super(NMTLossCompute, self).__init__(tgt_vocab_size, padding_idx)
        self.tgt_vocab_size = tgt_vocab_size
        self.padding_idx = padding_idx
        weight = torch.ones(tgt_vocab_size)
        weight[self.padding_idx] = 0
        self.criterion = nn.CrossEntropyLoss(weight, size_average=False)


    def forward(self, logits, target, length):
        return self.compute_loss(self, logits, target, length)

    def compute_loss(self, logits, target, length):
        length = Variable(torch.LongTensor(length)).cuda()
        logits = self.bottle(logits)
        target = self.bottle(target)
        loss = self.criterion(logits,target)
        loss = loss.sum() / length.float().sum()
        return  loss

        return loss, stats        

    def bottle(self, v):
        return v.view(-1, v.size(2))

    def unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))        