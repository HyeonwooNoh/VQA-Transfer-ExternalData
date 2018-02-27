import h5py
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from util import log

GLOVE_EMBEDDING_PATH = 'data/preprocessed/glove.6B.300d.hdf5'


def language_encoder(seq, seq_len, dim=384, scope='language_encoder', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if not reuse: log.warning(scope.name)
        cell = rnn.BasicLSTMCell(num_units=dim, state_is_tuple=True)
        _, final_state = tf.nn.dynamic_rnn(
            cell=cell, dtype=tf.float32, sequence_length=seq_len,
            inputs=seq)
        return final_state.h


def glove_embedding(seq, scope='glove_embedding', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if not reuse: log.warning(scope.name)
        with h5py.File(GLOVE_EMBEDDING_PATH, 'r') as f:
            fixed = tf.constant(f['param'].value.transpose())
        learn = tf.get_variable(
            name='learn', shape=[3, 300],
            initializer=tf.random_uniform_initializer(
                minval=-0.01, maxval=0.01))
        embed_map = tf.concat([fixed, learn], axis=0)
        embed = tf.nn.embedding_lookup(embed_map, seq)
        return embed
