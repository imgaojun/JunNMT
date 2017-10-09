import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from modules.Encoder import EncoderGRU
from modules.Decoder import AttnDecoderGRU



class NMTModel(nn.Module):
    def __init__(self, hparams):
        super(NMTModel, self).__init__()

    def forward(self):
        