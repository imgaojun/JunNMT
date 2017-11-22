import yaml
import torch.utils.data as data
import codecs
import math
import os
import sys, time

class HParams(object):
    def __init__(self, **entries): 
        self.__dict__.update(entries)   


def load_hparams(config_file):
    with codecs.open(config_file, 'r', encoding='utf8') as f:
        configs = yaml.load(f)
        hparams = HParams(**configs)
        return hparams

def print_hparams(hparams):
    for k,v in hparams.items():
        print(k,v)
        
def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def latest_checkpoint(model_dir):
    cnpt_file = os.path.join(model_dir,'checkpoint')
    cnpt = open(cnpt_file,'r').readline().strip().split(':')[-1]
    cnpt = os.path.join(model_dir,cnpt)
    return cnpt

 
class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 1 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 1

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        self.i += 1

    def close(self, words='done'):
        print('')
        print(words)
        self.i = 1