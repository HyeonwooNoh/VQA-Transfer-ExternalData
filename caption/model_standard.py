import cPickle
import h5py
import os
import numpy as np
import tensorflow as tf

from util import box_utils, log
from vlmap import modules

tc = tf.nn.rnn_cell

W_DIM = 300  # Word dimension
L_DIM = 1024 # Language dimension
V_DIM = 1024
R_DIM = 1024 # RNN dimension
ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'


class Model(object):

    def __init__(self, batch, config, is_train=True):
        self.batch = batch
        self.config = config
        self.image_dir = config.image_dir
        self.is_train = is_train

        # word_weight_dir is only for answer accuracy visualization
        self.word_weight_dir = getattr(config, 'vlmap_word_weight_dir', None)
        if self.word_weight_dir is None:
            log.warn('word_weight_dir is None')

        self.losses = {}
        self.report = {}
        self.mid_result = {}
        self.output = {}
        self.heavy_output = {}
        self.vis_image = {}

        self.vocab = cPickle.load(open(config.vocab_path, 'rb'))
        self.answer_dict = cPickle.load(open(
            os.path.join(config.tf_record_dir, 'answer_dict.pkl'), 'rb'))
        self.num_answer = len(self.answer_dict['vocab'])
        self.num_train_answer = self.answer_dict['num_train_answer']
        self.train_answer_mask = tf.expand_dims(tf.sequence_mask(
            self.num_train_answer, maxlen=self.num_answer, dtype=tf.float32),
            axis=0)
        self.test_answer_mask = 1.0 - self.train_answer_mask
        self.obj_answer_mask = tf.expand_dims(
            tf.constant(self.answer_dict['is_object'], dtype=tf.float32),
            axis=0)
        self.attr_answer_mask = tf.expand_dims(
            tf.constant(self.answer_dict['is_attribute'], dtype=tf.float32),
            axis=0)

        self.glove_map = modules.LearnGloVe(self.vocab)
        self.answer_exist_mask = modules.AnswerExistMask(
            self.answer_dict, self.word_weight_dir)

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
            if var.name.split('/')[0] == 'encode_L':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'GloVe':
                transfer_vars.append(var)
        return transfer_vars

    def get_enc_I_param_path(self):
        return ENC_I_PARAM_PATH

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
            # [B, # of box, dim]
            V_ft.set_shape([None, self.max_box_num, self.vfeat_dim])
            num_V_ft = tf.gather(self.num_boxes, self.batch['image_idx'],
                                 name='gather_num_V_ft', axis=0)
            self.mid_result['num_V_ft'] = num_V_ft
            normal_boxes = tf.gather(self.normal_boxes, self.batch['image_idx'],
                                     name='gather_normal_boxes', axis=0)
            self.mid_result['normal_boxes'] = normal_boxes

        log.warning('v_linear_v')
        # [B, # of box, V_DIM]
        v_linear_v = modules.fc_layer(
            V_ft, V_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='v_linear_v')

        """
        Average pooling
        """
        # [B, # of box, V_DIM] -> [B, V_DIM]
        avg_pooled_V_ft tf.reduce_mean(V_ft, 1, keepdims=False)

        # [B, R_DIM * 2]
        c_0_h_0 = modules.fc_layer(
            avg_pooled_V_ft, R_DIM * 2, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='c_0_h_0')

        # c_0, h_0: [B, R_DIM]
        c_0, h_0 = tf.split(c_0_h_0, [R_DIM, R_DIM], 1)
        state_in = tc.LSTMStateTuple(c_0, h_0)

        self.att_lstm = tc.BasicLSTMCell(R_DIM)
        self.lang_lstm = tc.BasicLSTMCell(R_DIM)

        lstm_out, lstm_state = tf.nn.dynamic_rnn(
            self.lang_lstm,
            # [batch_size, max_time, ...]
            lstm_in,
            # [batch_size, cell.state_size]
            initial_state=state_in,
            time_major=False)

        lstm_c, lstm_h = lstm_state

        # TODO(taehoon): time should be shifted

        # gate [batch_size, max_time, 1]
        gate = modules.fc_layer(
            lstm_h, 1, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.sigmoid, is_training=self.is_train,
            scope='gate')
        # visual sentinel [batch_size, max_time, R_DIM]
        sentinel = gate * tf.nn.tanh(lstm_c)

        sentinel_linear = modules.fc_layer(
            sentinel, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='sentinel_linear')
        h_linear = modules.fc_layer(
            lstm_h, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='h_linear')

        logit = tf.nn.tanh(tf.concat([state_t, lstm_h]))

        """
        Answer classification
        """
        # perform two layer feature encoding and predict output
        with tf.variable_scope('reasoning') as scope:
            log.warning(scope.name)
            # [bs, L_DIM]
            log.warning('pooled_linear_l')
            pooled_linear_l = modules.fc_layer(
                pooled_V_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
                activation_fn=tf.nn.relu, is_training=self.is_train,
                scope='pooled_linear_l')

            log.warning('q_linear_l')
            q_linear_l = modules.fc_layer(
                q_L_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
                activation_fn=tf.nn.relu, is_training=self.is_train,
                scope='q_linear_l')

            joint = modules.fc_layer(
                pooled_linear_l * q_linear_l, 2048,
                use_bias=True, use_bn=False, use_ln=True,
                activation_fn=tf.nn.relu, is_training=self.is_train, scope='joint_fc')
            joint = tf.nn.dropout(joint, 0.5)

            logit = modules.fc_layer(
                joint, self.num_answer,
                use_bias=True, use_bn=False, use_ln=False,
                activation_fn=None, is_training=self.is_train, scope='classifier')
        self.output['logit'] = logit

        """
        Compute loss and accuracy
        """
        with tf.name_scope('loss'):
            answer_target = self.batch['answer_target']
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=answer_target, logits=logit)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
            pred = tf.cast(tf.argmax(logit, axis=-1), dtype=tf.int32)
            one_hot_pred = tf.one_hot(pred, depth=self.num_answer,
                                      dtype=tf.float32)
            self.output['pred'] = pred
            all_score = tf.reduce_sum(one_hot_pred * answer_target, axis=-1)
            max_train_score = tf.reduce_max(
                answer_target * self.train_answer_mask, axis=-1)
            self.output['all_score'] = all_score
            self.output['max_train_score'] = max_train_score

            acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target, axis=-1))
            exist_acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target * self.answer_exist_mask,
                              axis=-1))
            test_acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target * self.test_answer_mask,
                              axis=-1))
            test_obj_acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target * self.test_answer_mask *
                              self.obj_answer_mask, axis=-1))
            test_attr_acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target * self.test_answer_mask *
                              self.attr_answer_mask, axis=-1))
            train_exist_acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target * self.answer_exist_mask *
                              self.train_answer_mask,
                              axis=-1))
            max_exist_answer_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.answer_exist_mask, axis=-1))
            max_train_exist_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.answer_exist_mask *
                              self.train_answer_mask, axis=-1))
            test_obj_max_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.test_answer_mask *
                              self.obj_answer_mask, axis=-1))
            test_attr_max_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.test_answer_mask *
                              self.attr_answer_mask, axis=-1))
            test_max_answer_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.test_answer_mask, axis=-1))
            test_max_exist_answer_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.answer_exist_mask *
                              self.test_answer_mask, axis=-1))
            normal_test_obj_acc = tf.where(
                tf.equal(test_obj_max_acc, 0),
                test_obj_max_acc,
                test_obj_acc / test_obj_max_acc)
            normal_test_attr_acc = tf.where(
                tf.equal(test_attr_max_acc, 0),
                test_attr_max_acc,
                test_attr_acc / test_attr_max_acc)
            normal_train_exist_acc = tf.where(
                tf.equal(max_train_exist_acc, 0),
                max_train_exist_acc,
                train_exist_acc / max_train_exist_acc)
            normal_exist_acc = tf.where(
                tf.equal(max_exist_answer_acc, 0),
                max_exist_answer_acc,
                exist_acc / max_exist_answer_acc)
            normal_test_acc = tf.where(
                tf.equal(test_max_answer_acc, 0),
                test_max_answer_acc,
                test_acc / test_max_answer_acc)

            self.mid_result['pred'] = pred

            self.losses['answer'] = loss
            self.report['answer_train_loss'] = loss
            self.report['answer_report_loss'] = loss
            self.report['answer_acc'] = acc
            self.report['exist_acc'] = exist_acc
            self.report['test_acc'] = test_acc
            self.report['normal_test_acc'] = normal_test_acc
            self.report['normal_test_object_acc'] = normal_test_obj_acc
            self.report['normal_test_attribute_acc'] = normal_test_attr_acc
            self.report['normal_exist_acc'] = normal_exist_acc
            self.report['normal_train_exist_acc'] = normal_train_exist_acc
            self.report['max_exist_acc'] = max_exist_answer_acc
            self.report['test_max_acc'] = test_max_answer_acc
            self.report['test_max_exist_acc'] = test_max_exist_answer_acc

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
