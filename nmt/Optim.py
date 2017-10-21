import torch.optim as optim
from torch.nn.utils import clip_grad_norm


class Optim(object):

    def __init__(self, method, lr, max_grad_norm,
                 lr_decay=1, start_decay_at=None,
                 beta1=0.9, beta2=0.98):

        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.betas = [beta1, beta2]

    def set_parameters(self, params):
        self.params = [p for p in params if p.requires_grad]
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr,
                                        betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def step(self):
        "Compute gradients norm."
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()