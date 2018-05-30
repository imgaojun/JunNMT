import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from nmt.Trainer import Statistics
import nmt.IO


class NMTLossCompute(nn.Module):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab):
        super(NMTLossCompute, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[nmt.IO.PAD_WORD]
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def make_shard_state(self, batch, output):
        """ See base class for args description. """
        return {
            "output": output,
            "target": batch.tgt[1:],
            }   

    def compute_loss(self, batch, output, target):

        scores = self.generator(self.bottle(output))
        target = target.view(-1)
        loss = self.criterion(scores,target)

        loss_data = loss.data.clone()
        stats = self.stats(loss_data, scores.data, target.data)
        return  loss, stats

    def compute_train_loss(self, batch, output):
        """
        Compute the loss in shards for efficiency.
        """
        batch_stats = Statistics()
        shard_state = self.make_shard_state(batch, output)
        loss, stats = self.compute_loss(batch, **shard_state)
        loss.div(batch.batch_size).backward()
        batch_stats.update(stats)

        return batch_stats       
        
    def compute_valid_loss(self, batch, output):
        """
        Compute the loss monolithically, not dividing into shards.
        """

        shard_state = self.make_shard_state(batch, output)
        _, batch_stats = self.compute_loss(batch, **shard_state)

        return batch_stats

    def stats(self, loss, scores, target):
        """
        Compute and return a Statistics object.
        Args:
            loss(Tensor): the loss computed by the loss criterion.
            scores(Tensor): a sequence of predict output with scores.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return Statistics(loss[0], non_padding.sum(), num_correct)

    def bottle(self, v):
        return v.view(-1, v.size(2))

    def unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))        