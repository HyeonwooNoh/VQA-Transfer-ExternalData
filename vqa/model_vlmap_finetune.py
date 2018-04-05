import cPickle
import h5py
import os
import numpy as np
import tensorflow as tf

from util import log
from vlmap import modules

W_DIM = 300  # Word dimension
L_DIM = 1024  # Language dimension
V_DIM = 1024


class Model(object):

    def __init__(self, batch, config, is_train=True):
        self.batch = batch
        self.config = config
        self.image_dir = config.image_dir
        self.is_train = is_train

        self.pretrained_param_path = config.pretrained_param_path
        if self.pretrained_param_path is None:
            raise ValueError('pretrained_param_path is mendatory')
        self.word_weight_dir = config.vlmap_word_weight_dir
        if self.word_weight_dir is None:
            raise ValueError('word_weight_dir is mendatory')

        self.losses = {}
        self.report = {}
        self.mid_result = {}
        self.vis_image = {}

        self.vocab = cPickle.load(open(config.vocab_path, 'rb'))
        self.answer_dict = cPickle.load(open(
            os.path.join(config.tf_record_dir, 'answer_dict.pkl'), 'rb'))
        self.num_answer = len(self.answer_dict['vocab'])
        self.num_train_answer = self.answer_dict['num_train_answer']
        self.train_answer_mask = tf.expand_dims(tf.sequence_mask(
            self.num_train_answer, maxlen=self.num_answer, dtype=tf.float32),
            axis=0)

        self.glove_map = modules.LearnGloVe(self.vocab)
        self.v_word_map = modules.WordWeightEmbed(
            self.vocab, self.word_weight_dir, 'v_word', scope='V_WordMap')

        log.infov('loading image features...')
        with h5py.File(config.vfeat_path, 'r') as f:
            self.features = np.array(f.get('image_features'))
            log.infov('feature done')
            self.spatials = np.array(f.get('spatial_features'))
            log.infov('spatials done')
            self.normal_boxes = np.array(f.get('normal_boxes'))
            log.infov('normal_boxes done')
            self.num_boxes = np.array(f.get('num_boxes'))
            log.infov('num_boxes done')
            self.max_box_num = int(f['data_info']['max_box_num'].value)
            self.vfeat_dim = int(f['data_info']['vfeat_dim'].value)
        log.infov('done')

        self.build()

    def filter_train_vars(self, trainable_vars):
        train_vars = []
        for var in trainable_vars:
            train_vars.append(var)
        return train_vars

    def filter_transfer_vars(self, all_vars):
        transfer_vars = []
        for var in all_vars:
            if var.name.split('/')[0] == 'v_word_fc':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'q_linear_v':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'v_linear_v':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'hadamard_attention':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'q_linear_l':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'pooled_linear_l':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'joint_fc':
                transfer_vars.append(var)
        return transfer_vars

    def build(self):
        """
        build network architecture and loss
        """

        """
        Visual features
        """
        with tf.device('/cpu:0'):
            def load_feature(image_idx):
                selected_features = np.take(self.features, image_idx, axis=0)
                return selected_features
            V_ft = tf.py_func(
                load_feature, inp=[self.batch['image_idx']], Tout=tf.float32,
                name='sample_features')
            V_ft.set_shape([None, self.max_box_num, self.vfeat_dim])
            num_V_ft = tf.gather(self.num_boxes, self.batch['image_idx'],
                                 name='gather_num_V_ft', axis=0)
            self.mid_result['num_V_ft'] = num_V_ft
            normal_boxes = tf.gather(self.normal_boxes, self.batch['image_idx'],
                                     name='gather_normal_boxes', axis=0)
            self.mid_result['normal_boxes'] = normal_boxes

        log.warning('v_linear_v')
        v_linear_v = modules.fc_layer(
            V_ft, V_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='v_linear_v')

        """
        Encode question
        """
        q_token, q_len = self.batch['q_intseq'], self.batch['q_intseq_len']
        q_embed = tf.nn.embedding_lookup(self.glove_map, q_token)
        q_L_map, q_L_ft = modules.encode_L_bidirection(
            q_embed, q_len, L_DIM, scope='encode_L_bi', cell_type='GRU')

        q_att_key = modules.fc_layer(  # [bs, len, L_DIM]
            q_L_map, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_att_key')

        q_att_query = modules.fc_layer(  # [bs, L_DIM]
            q_L_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_att_query')

        w_att_score = modules.hadamard_attention(
            q_att_key, q_len, q_att_query, use_ln=False, is_train=self.is_train,
            scope='word_attention')

        q_v_embed = tf.nn.embedding_lookup(self.v_word_map, q_token)
        q_v_ft = modules.fc_layer(  # [bs, len, L_DIM]
            q_v_embed, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='v_word_fc')

        pooled_q_v = modules.attention_pooling(q_v_ft, w_att_score)

        # [bs, V_DIM}
        log.warning('q_linear_v')
        q_linear_v = modules.fc_layer(
            pooled_q_v, V_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_v')

        """
        Perform attention
        """
        att_score = modules.hadamard_attention(v_linear_v, num_V_ft, q_linear_v,
                                               use_ln=False, is_train=self.is_train,
                                               scope='hadamard_attention')
        self.mid_result['att_score'] = att_score
        pooled_V_ft = modules.attention_pooling(V_ft, att_score)

        """
        Answer classification
        """
        log.warning('pooled_linear_l')
        pooled_linear_l = modules.fc_layer(
            pooled_V_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='pooled_linear_l')

        log.warning('q_linear_l')
        l_linear_l = modules.fc_layer(
            q_L_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_l')

        joint = modules.fc_layer(
            pooled_linear_l * l_linear_l, L_DIM * 2,
            use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train, scope='joint_fc')
        joint = tf.nn.dropout(joint, 0.5)

        logit = modules.WordWeightAnswer(
            joint, self.answer_dict, self.word_weight_dir,
            use_bias=True, is_training=self.is_train, scope='WordWeightAnswer')

        """
        Compute loss and accuracy
        """
        with tf.name_scope('loss'):
            answer_target = self.batch['answer_target']
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=answer_target, logits=logit)

            train_loss = tf.reduce_mean(tf.reduce_sum(
                loss * self.train_answer_mask, axis=-1))
            report_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
            pred = tf.cast(tf.argmax(logit, axis=-1), dtype=tf.int32)
            one_hot_pred = tf.one_hot(pred, depth=self.num_answer,
                                      dtype=tf.float32)
            acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target, axis=-1))

            self.mid_result['pred'] = pred

            self.losses['answer'] = train_loss
            self.report['answer_train_loss'] = train_loss
            self.report['answer_report_loss'] = report_loss
            self.report['answer_accuracy'] = acc

        """
        Prepare image summary
        """
        """
        with tf.name_scope('prepare_summary'):
            self.vis_image['image_attention_qa'] = self.visualize_vqa_result(
                self.batch['image_id'],
                self.mid_result['normal_boxes'], self.mid_result['num_V_ft'],
                self.mid_result['att_score'],
                self.batch['q_intseq'], self.batch['q_intseq_len'],
                self.batch['answer_target'], self.mid_result['pred'],
                max_batch_num=20, line_width=2)
        """

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
