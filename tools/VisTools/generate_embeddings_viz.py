#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import sys, os

def read_vecs(filename):
    words = []
    values = []
    for l in open(filename):
        t = l.split(" ")
        words.append(t[0])
        values.append([float(a) for a in t[1:]])
    return words, np.array(values)

def write_metadata(filename, words):
    with open(filename, 'w') as w:
        for word in words:
            w.write(word + "\n")

src_words, src_values = read_vecs(sys.argv[1] + "/src_embeddings.txt")
tgt_words, tgt_values = read_vecs(sys.argv[1] + "/tgt_embeddings.txt")

tf.reset_default_graph()
src_embedding_var = tf.Variable(src_values, name="src_embeddings")
tgt_embedding_var = tf.Variable(tgt_values, name="tgt_embeddings")
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    saver = tf.train.Saver()
    saver.save(session, "/tmp/model.ckpt", 1)

write_metadata("/tmp/src_metadata.tsv", src_words)
write_metadata("/tmp/tgt_metadata.tsv", tgt_words)

from tensorflow.contrib.tensorboard.plugins import projector
summary_writer = tf.summary.FileWriter("/tmp/")

config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = src_embedding_var.name
embedding.metadata_path = '/tmp/src_metadata.tsv'

embedding = config.embeddings.add()
embedding.tensor_name = tgt_embedding_var.name
embedding.metadata_path = '/tmp/tgt_metadata.tsv'

projector.visualize_embeddings(summary_writer, config)
os.system("tensorboard --log=/tmp/")
