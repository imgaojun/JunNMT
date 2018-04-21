import yaml
import torch.utils.data as data
import codecs
import math
import os
import sys, time
import nmt
class HParams(object):
    def __init__(self, **entries): 
        self.__dict__.update(entries)   


def load_hparams(config_file):
    with codecs.open(config_file, 'r', encoding='utf8') as f:
        configs = yaml.load(f)
        hparams = HParams(**configs)
        return hparams

        
def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def latest_checkpoint(model_dir):
    
    cnpt_file = os.path.join(model_dir,'checkpoint')
    try:
        cnpt = open(cnpt_file,'r').readline().strip().split(':')[-1]
    except:
        return None
    cnpt = os.path.join(model_dir,cnpt)
    return cnpt
