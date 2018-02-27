import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from util import log
from vlmap import modules

L_DIM = 384  # Language dimension
ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'
ENC_I_R_MEAN = 123.68
ENC_I_G_MEAN = 116.78
ENC_I_B_MEAN = 103.94


class Model(object):

    def __init__(self, batches, config, global_step=None, is_train=True):
        self.batches = batches
        self.config = config
        self.global_step = global_step

        self.batch_size = config.batch_size
        self.object_num_k = config.object_num_k
        self.object_max_name_len = config.object_max_name_len

        self.build(is_train=is_train)

    def filter_vars(self, all_vars):
        enc_I_vars = []
        learn_vars = []
        for var in all_vars:
            if var.name.split('/')[0] == 'resnet_v1_50':
                enc_I_vars.append(var)
            else:
                learn_vars.append(var)
        return enc_I_vars, learn_vars

    def get_enc_I_param_path(self):
        return ENC_I_PARAM_PATH

    def build(self, is_train=True):

        """
        Pre-trained model parameter is available here:
        https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
        """
        with tf.name_scope('enc_I_preprocess'):
            channels = tf.split(axis=3, num_or_size_splits=3,
                                value=self.batches['object']['image'])
            for i, mean in enumerate([ENC_I_R_MEAN, ENC_I_G_MEAN, ENC_I_B_MEAN]):
                channels[i] -= mean
            processed_I = tf.concat(axis=3, values=channels)

        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            enc_I, _ = nets.resnet_v1.resnet_v1_50(
                processed_I,
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

        with tf.name_scope('Accuracy'):
            label_token = tf.argmax(labels, axis=-1)
            logit_token = tf.argmax(logits, axis=-1)
            acc = tf.reduce_mean(tf.to_float(
                tf.equal(label_token, logit_token)))

        with tf.name_scope('Summary'):
            image = self.batches['object']['image'] / 255.0

            def visualize_prediction(logit, label):
                prob = tf.nn.softmax(logit, dim=-1)
                prob_image = tf.expand_dims(prob, axis=-1)
                label_image = tf.expand_dims(label, axis=-1)
                dummy = tf.zeros_like(label_image)
                pred_image = tf.clip_by_value(
                    tf.concat([prob_image, label_image, dummy], axis=-1),
                    0, 1)
                pred_image = tf.tile(pred_image, [10, 1, 1])
                return tf.expand_dims(pred_image, axis=0)
            pred_image = visualize_prediction(logits, labels)

        tf.summary.scalar('train/loss', self.loss, collections=['train'])
        tf.summary.scalar('val/loss', self.loss, collections=['val'])

        tf.summary.scalar('train/accuracy', acc, collections=['train'])
        tf.summary.scalar('val/accuracy', acc, collections=['val'])

        tf.summary.image('train_image', image, collections=['train'])
        tf.summary.image('train_prediction_image', pred_image,
                         collections=['train'])
        tf.summary.image('val_image', image, collections=['val'])
        tf.summary.image('val_prediction_image', pred_image,
                         collections=['val'])
