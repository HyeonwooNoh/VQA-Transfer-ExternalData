import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from util import log
from vlmap import modules

L_DIM = 384  # Language dimension


class Model(object):

    def __init__(self, batches, config, global_step=None, is_train=True):
        self.batches = batches
        self.config = config
        self.global_step = global_step

        self.batch_size = config.batch_size
        self.object_num_k = config.object_num_k
        self.object_max_name_len = config.object_max_name_len

        self.build(is_train=is_train)

    def build(self, is_train=True):

        """
        Pre-trained model parameter is available here:
        https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
        """

        enc_I, _ = nets.resnet_v1.resnet_v1_50(
            self.batches['object']['image'],
            is_training=False,
            global_pool=True,
            output_stride=None,
            reuse=None,
            scope='resnet_v1_50')
        enc_I = tf.stop_gradient(tf.squeeze(enc_I, axis=[1, 2]))

        embed_seq = modules.glove_embedding(
            self.batches['object']['objects'],
            scope='glove_embedding', reuse=False)
        enc_L_flat = modules.language_encoder(
            tf.reshape(embed_seq, [-1, self.object_max_name_len, 300]),
            tf.reshape(self.batches['object']['objects_len'], [-1]),
            L_DIM, scope='language_encoder', reuse=False)
        enc_L = tf.reshape(enc_L_flat,
                           [-1, self.object_num_k, L_DIM])

        with tf.variable_scope('Classifier') as scope:
            log.warning(scope.name)
            map_I = layers.fully_connected(
                enc_I, L_DIM, activation_fn=None, biases_initializer=None,
                reuse=None, trainable=True, scope='map_I')
            tiled_map_I = tf.tile(tf.expand_dims(map_I, axis=1),
                                  [1, self.object_num_k, 1])
            bias = tf.get_variable(name='bias', shape=(),
                                   initializer=tf.zeros_initializer())
            logits = tf.reduce_sum(tiled_map_I * enc_L, axis=-1) + bias

        with tf.name_scope('Loss'):
            labels = self.batches['object']['ground_truth']
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(labels), logits=logits)
            self.loss = tf.reduce_mean(cross_entropy)

        tf.summary.scalar('train/loss', self.loss, collections=['train'])
        tf.summary.scalar('val/loss', self.loss, collections=['val'])
