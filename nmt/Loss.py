import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable




class NMTLossCompute(nn.Module):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, tgt_vocab_size, padding_idx):
        super(NMTLossCompute, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.padding_idx = padding_idx
        weight = torch.ones(tgt_vocab_size)
        weight[self.padding_idx] = 0
        self.criterion = nn.CrossEntropyLoss(weight, size_average=False)


    def compute_loss(self, output, target, length):
        length = Variable(torch.LongTensor(length)).cuda()
        batch_size = output.size(0)
        output = self.bottle(output)
        target = target.view(-1)
        loss = self.criterion(output,target)
        # loss = loss.sum()/length.float().sum()
        loss = loss.div(batch_size)
        return  loss

    # def make_shard_state(self, batch, output, range_, attns=None):
    #     """ See base class for args description. """
    #     return {
    #         "output": output,
    #         "target": batch.tgt[range_[0] + 1: range_[1]],
    #     }


    # def shard_compute_loss(self,output, target, shard_size):
    #     time_steps = output.size(1)
    #     for shard in range(0,time_steps,shard_size):
    #         loss = 

    # def monolithic_compute_loss(self):

    def bottle(self, v):
        return v.view(-1, v.size(2))

    def unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))        


class MaskCrossEntropy(object):

    def _sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1)
                            .expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand


    def compute_loss(self, logits, target, length):
        length = Variable(torch.LongTensor(length)).cuda()
        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """

        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = functional.log_softmax(logits_flat)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        mask = self._sequence_mask(sequence_length=length, max_len=target.size(1))
        losses = losses * mask.float()
        loss = losses.sum() / length.float().sum()
        return loss         