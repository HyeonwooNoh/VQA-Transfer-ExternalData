import h5py
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from util import log
from vlmap import modules

TOP_K = 5
L_DIM = 384  # Language dimension
MAP_DIM = 384
V_DIM = 384
ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'


class Model(object):

    def __init__(self, batches, config, is_train=True):
        self.batches = batches
        self.config = config

        self.report = {}
        self.output = {}

        self.batch_size = config.batch_size
        self.object_num_k = config.object_num_k
        self.object_max_name_len = config.object_max_name_len

        self.used_wordset_path = config.used_wordset_path

        # model parameters
        self.finetune_enc_I = config.finetune_enc_I
        self.no_finetune_enc_L = config.no_finetune_enc_L

        self.build(is_train=is_train)

    def filter_vars(self, all_vars):
        enc_I_vars = []
        learn_vars = []
        for var in all_vars:
            if var.name.split('/')[0] == 'resnet_v1_50':
                enc_I_vars.append(var)
                if self.finetune_enc_I:
                    learn_vars.append(var)
            else:
                learn_vars.append(var)
        return enc_I_vars, learn_vars

    def get_enc_I_param_path(self):
        return ENC_I_PARAM_PATH

    def visualize_word_prediction(self, logit, label):
        prob = tf.nn.softmax(logit, dim=-1)
        prob_image = tf.expand_dims(prob, axis=-1)
        label_image = tf.expand_dims(label, axis=-1)
        dummy = tf.zeros_like(label_image)
        pred_image = tf.clip_by_value(
            tf.concat([prob_image, label_image, dummy], axis=-1),
            0, 1)
        pred_image = tf.reshape(
            tf.tile(pred_image, [1, 10, 1]), [-1, self.object_num_k, 3])
        return tf.expand_dims(pred_image, axis=0)

    def visualize_word_prediction_text(self, logits, labels, names):
        label_token = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        _, top_k_pred = tf.nn.top_k(logits, k=TOP_K)
        batch_range = tf.expand_dims(
            tf.range(0, tf.shape(label_token)[0], delta=1), axis=1)
        range_label_token = tf.concat(
            [batch_range, tf.expand_dims(label_token, axis=1)], axis=1)
        label_name = tf.gather_nd(names, range_label_token)
        top_k_preds = tf.split(axis=-1, num_or_size_splits=TOP_K,
                               value=top_k_pred)
        pred_names = []
        for i in range(TOP_K):
            range_top_k_pred = tf.concat(
                [batch_range, top_k_preds[i]], axis=1)
            pred_names.append(tf.gather_nd(names, range_top_k_pred))
        string_list = ['gt: ', label_name]
        for i in range(TOP_K):
            string_list.extend([', pred({}): '.format(i), pred_names[i]])
        pred_string = tf.string_join(string_list)
        return pred_string

    def word_prediction_loss(self, logits, labels):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(labels), logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        # Accuracy
        label_token = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        logit_token = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        acc = tf.reduce_mean(tf.to_float(
            tf.equal(label_token, logit_token)))
        _, top_k_pred = tf.nn.top_k(logits, k=TOP_K)
        k_label_token = tf.tile(
            tf.expand_dims(label_token, axis=1), [1, TOP_K])
        top_k_acc = tf.reduce_mean(tf.to_float(tf.reduce_any(
            tf.equal(k_label_token, top_k_pred), axis=1)))
        return loss, acc, top_k_acc

    def add_classification_summary(self, loss, acc, top_k_acc, image,
                                   pred_image, pred_string, name='default'):
        tf.summary.scalar('train-{}/loss'.format(name),
                          loss, collections=['train'])
        tf.summary.scalar('val-{}/loss'.format(name), loss, collections=['val'])

        tf.summary.scalar('train-{}/accuracy'.format(name),
                          acc, collections=['train'])
        tf.summary.scalar('val-{}/accuracy'.format(name),
                          acc, collections=['val'])

        tf.summary.scalar('train-{}/top_{}_acc'.format(name, TOP_K),
                          top_k_acc, collections=['train'])
        tf.summary.scalar('val-{}/top_{}_acc'.format(name, TOP_K),
                          top_k_acc, collections=['val'])
        tf.summary.image('train-{}_image'.format(name),
                         image, collections=['train'])
        tf.summary.image('train-{}_prediction_image'.format(name), pred_image,
                         collections=['train'])
        tf.summary.image('val-{}_image'.format(name),
                         image, collections=['val'])
        tf.summary.image('val-{}_prediction_image'.format(name), pred_image,
                         collections=['val'])
        tf.summary.text('train-{}_pred_string'.format(name),
                        pred_string, collections=['train'])
        tf.summary.text('val-{}_pred_string'.format(name),
                        pred_string, collections=['val'])

    def build_object(self, is_train=True, add_summary=True):
        # Object class encoder
        enc_I = modules.encode_I(self.batches['object']['image'],
                                 is_train=self.finetune_enc_I,
                                 reuse=False)
        if not self.finetune_enc_I: enc_I = tf.stop_gradient(enc_I)
        feat_V = modules.I2V(enc_I, MAP_DIM, V_DIM, scope='I2V',
                             is_train=is_train, reuse=False)
        embed_seq = modules.glove_embedding(
            self.batches['object']['objects'],
            scope='glove_embedding', reuse=False)
        enc_L_flat = modules.language_encoder(
            tf.reshape(embed_seq, [-1, self.object_max_name_len, 300]),
            tf.reshape(self.batches['object']['objects_len'], [-1]),
            L_DIM, scope='language_encoder', reuse=False)
        enc_L = tf.reshape(enc_L_flat,
                           [-1, self.object_num_k, L_DIM])
        if self.no_finetune_enc_L: enc_L = tf.stop_gradient(enc_L)
        map_V = modules.L2V(enc_L, MAP_DIM, V_DIM, is_train=is_train,
                            scope='L2V', reuse=False)
        logits = modules.batch_word_classifier(
            feat_V, map_V, scope='object_classifier', reuse=False)

        with tf.name_scope('ObjectClassLoss'):
            labels = self.batches['object']['ground_truth']
            loss, acc, top_k_acc = self.word_prediction_loss(
                logits, labels)
            self.report.update({'object-loss': loss, 'object-acc': acc,
                                'object-top_{}_acc'.format(TOP_K): top_k_acc})

        with tf.name_scope('Summary'):
            image = self.batches['object']['image'] / 255.0
            pred_image = self.visualize_word_prediction(logits, labels)
            pred_string = self.visualize_word_prediction_text(
                logits, labels, self.batches['object']['objects_name'])
            self.output.update({'object-image': image,
                                'object-prediction_string': pred_string})

        if add_summary:
            self.add_classification_summary(loss, acc, top_k_acc, image,
                                            pred_image, pred_string,
                                            name='object')
        return loss

    def build(self, is_train=True):
        object_loss = self.build_object(is_train=is_train, add_summary=True)

        self.loss = 0
        self.loss += object_loss

        tf.summary.scalar('train/loss', self.loss, collections=['train'])
        tf.summary.scalar('val/loss', self.loss, collections=['val'])

        self.report['loss'] = self.loss
