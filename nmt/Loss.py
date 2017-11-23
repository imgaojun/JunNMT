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

    def sharded_compute_loss(self, batch, output, shard_size):
        """
        Compute the loss in shards for efficiency.
        """
        batch_stats = Statistics()
        shard_state = self.make_shard_state(batch, output)

        for shard in shards(shard_state, shard_size):
            loss, stats = self.compute_loss(batch, **shard)
            loss.div(batch.batch_size).backward()
            batch_stats.update(stats)

        return batch_stats

    def compute_valid_loss(self, logits, target):

        logits = self.bottle(logits)
        target = target.view(-1)
        loss = self.criterion(logits,target)
        loss_data = loss.data.clone()
        stats = self.stats(loss_data, logits.data, target.data)
        return  stats        
        
    def monolithic_compute_loss(self, batch, output):
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


def filter_shard_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute.make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.
    Yields:
        Each yielded shard is a dict.
    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))
        
        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, torch.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)