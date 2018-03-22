import json
import h5py
import os
import numpy as np
import tensorflow as tf

from util import log
from vlmap import modules

W_DIM = 300  # Word dimension
L_DIM = 512  # Language dimension
MAP_DIM = 512
V_DIM = 512
ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'

ROI_SZ = 5

# Visualization
VIS_NUMBOX = 5
LINE_WIDTH = 2


class Model(object):

    def __init__(self, batch, config, is_train=True):
        self.batch = batch
        self.config = config
        self.data_cfg = config.dataset_config

        self.losses = {}
        self.report = {}
        self.mid_result = {}
        self.vis_image = {}

        self.vocab = json.load(open(config.vocab_path, 'r'))
        self.glove_map = modules.GloVe_vocab(self.vocab)

        # model parameters
        self.ft_vlmap = config.ft_vlmap

        # answer candidates
        with h5py.File(os.path.join(config.dataset_path, 'data.hdf5'), 'r') as f:
            self.answer_intseq = tf.constant(
                f['data_info']['intseq_ans'].value)  # [num_answer, len]
            self.answer_intseq_len = tf.constant(
                f['data_info']['intseq_ans_len'].value)  # [num_answer]

        self.build(is_train=is_train)

    def filter_train_vars(self, trainable_vars):
        train_vars = []
        for var in trainable_vars:
            if var.name.split('/')[0] == 'V2L':
                if self.ft_vlmap:
                    train_vars.append(var)
            elif var.name.split('/')[0] == 'L2V':
                if self.ft_vlmap:
                    train_vars.append(var)
            else:
                train_vars.append(var)
        return train_vars

    def filter_transfer_vars(self, all_vars):
        transfer_vars = []
        for var in all_vars:
            if var.name.split('/')[0] == 'V2L':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'L2V':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'encode_L':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'GloVe':
                transfer_vars.append(var)
        return transfer_vars

    def get_enc_I_param_path(self):
        return ENC_I_PARAM_PATH

    def build(self, is_train=True):
        """
        build network architecture and loss
        """

        """
        Visual features
        """
        V_ft = self.batch['V_ft']  # [bs, num_box, V_DIM]
        num_V_ft = self.batch['num_box']  # [bs]

        """
        Encode question
        """
        q_embed = tf.nn.embedding_lookup(self.glove_map, self.batch['q_intseq'])
        # [bs, L_DIM]
        q_L_ft = modules.encode_L(q_embed, self.batch['q_intseq_len'], L_DIM)

        # [bs, V_DIM}
        q_map_V = modules.L2V(q_L_ft, MAP_DIM, V_DIM, is_train=is_train)

        """
        Perform attention
        """
        att_score = modules.attention(V_ft, num_V_ft, q_map_V)
        pooled_V_ft = modules.attention_pooling(V_ft, att_score)
        # [bs, L_DIM]
        pooled_map_L, _ = modules.V2L(pooled_V_ft, MAP_DIM, L_DIM,
                                      is_train=is_train)
        """
        Answer classification
        """
        answer_embed = tf.nn.embedding_lookup(self.glove_map, self.answer_intseq)
        # [num_answer, L_DIM]
        answer_ft = modules.encode_L(answer_embed, self.answer_intseq_len, L_DIM)

        # perform two layer feature encoding and predict output
        with tf.variable_scope('reasoning') as scope:
            log.warning(scope.name)
            # layer 1
            # answer_layer1: [1, num_answer, L_DIM]
            # pooled_layer1: [bs, 1, L_DIM]
            # q_layer1: [bs, 1, L_DIM]
            # layer1: [bs, num_answer, L_DIM]
            answer_layer1 = modules.fc_layer(
                answer_ft, L_DIM, use_bias=False, use_bn=False,
                activation_fn=None, is_training=is_train, scope='answer_layer1')
            answer_layer1 = tf.expand_dims(answer_layer1, axis=0)
            pooled_layer1 = modules.fc_layer(
                pooled_map_L, L_DIM, use_bias=False, use_bn=False,
                activation_fn=None, is_training=is_train, scope='pooled_layer1')
            pooled_layer1 = tf.expand_dims(pooled_layer1, axis=1)
            q_layer1 = modules.fc_layer(
                q_L_ft, L_DIM, use_bias=True, use_bn=False,
                activation_fn=None, is_training=is_train, scope='q_layer1')
            q_layer1 = tf.expand_dims(q_layer1, axis=1)
            layer1 = tf.tanh(answer_layer1 + pooled_layer1 + q_layer1)

            logit = modules.fc_layer(
                layer1, 1, use_bias=True, use_bn=False,
                activation_fn=None, is_training=is_train, scope='classifier')
            logit = tf.squeeze(logit, axis=-1)  # [bs, num_answer]

            # concat inf
            inf = tf.as_dtype(logit.dtype).as_numpy_dtype(-np.inf)
            inf_column = tf.ones(
                [tf.shape(logit)[0], 1], dtype=logit.dtype) * inf
            logit = tf.concat([logit, inf_column], axis=1)  # [bs, num_answer + 1]

        """
        Compute loss and accuracy
        """
        with tf.name_scope('loss'):
            label = self.batch['answer_id']
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label, logits=logit))
            pred = tf.argmax(logit, axis=-1)
            acc = tf.reduce_mean(tf.to_float(pred == label))

            self.losses['answer'] = loss
            self.report['answer_loss'] = loss
            self.report['answer_accuracy'] = acc

        self.loss = self.losses['answer']

        # scalar summary
        for key, val in self.report.items():
            tf.summary.scalar('train/{}'.format(key), val,
                              collections=['heavy_train', 'train'])
            tf.summary.scalar('val/{}'.format(key), val,
                              collections=['heavy_val', 'val'])
            tf.summary.scalar('testval/{}'.format(key), val,
                              collections=['heavy_testval', 'testval'])

        # image summary
        for key, val in self.vis_image.items():
            tf.summary.image('train-{}'.format(key), val, max_outputs=10,
                             collections=['heavy_train'])
            tf.summary.image('val-{}'.format(key), val, max_outputs=10,
                             collections=['heavy_val'])
            tf.summary.image('testval-{}'.format(key), val, max_outputs=10,
                             collections=['heavy_testval'])

        return self.loss
