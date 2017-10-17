import os
import codecs
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--src_in", type=str)
parser.add_argument("--tgt_in", type=str)
parser.add_argument("--ref_dir", type=str)

args = parser.parse_args()