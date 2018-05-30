import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Optim(object):

    def __init__(self, method, lr, max_grad_norm,
                 lr_decay=1, weight_decay=0, 
                 start_decay_at=None,
                 beta1=0.9, beta2=0.98):
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]

    def _setRate(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def set_parameters(self, params):
        self.params = [p for p in params if p.requires_grad]
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr,
                                        betas=self.betas, eps=1e-9, 
                                        weight_decay=self.weight_decay)                                     
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def step(self):
        "Compute gradients norm."
        self._step += 1

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.step()

    def updateLearningRate(self, ppl, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl
        self.optimizer.param_groups[0]['lr'] = self.lr        