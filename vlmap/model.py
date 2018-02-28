import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from util import log
from vlmap import modules

TOP_K = 5
L_DIM = 384  # Language dimension
MAP_DIM = 384
ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'
ENC_I_R_MEAN = 123.68
ENC_I_G_MEAN = 116.78
ENC_I_B_MEAN = 103.94


class Model(object):

    def __init__(self, batches, config, is_train=True):
        self.batches = batches
        self.config = config

        self.batch_size = config.batch_size
        self.object_num_k = config.object_num_k
        self.object_max_name_len = config.object_max_name_len

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
                is_training=self.finetune_enc_I,
                global_pool=True,
                output_stride=None,
                reuse=None,
                scope='resnet_v1_50')
            enc_I = tf.squeeze(enc_I, axis=[1, 2])
            if not self.finetune_enc_I:
                enc_I = tf.stop_gradient(enc_I)

        with tf.variable_scope('I2V') as scope:
            log.warning(scope.name)
            feat_V = modules.fc_layer(
                enc_I, MAP_DIM, use_bias=False, use_bn=False,
                activation_fn=None, is_training=is_train,
                scope='Linear', reuse=False)

        embed_seq = modules.glove_embedding(
            self.batches['object']['objects'],
            scope='glove_embedding', reuse=False)
        enc_L_flat = modules.language_encoder(
            tf.reshape(embed_seq, [-1, self.object_max_name_len, 300]),
            tf.reshape(self.batches['object']['objects_len'], [-1]),
            L_DIM, scope='language_encoder', reuse=False)
        enc_L = tf.reshape(enc_L_flat,
                           [-1, self.object_num_k, L_DIM])
        if self.no_finetune_enc_L:
            enc_L = tf.stop_gradient(enc_L)

        with tf.variable_scope('L2V') as scope:
            log.warning(scope.name)
            map_V = modules.fc_layer(
                enc_L, MAP_DIM, use_bias=False, use_bn=True,
                activation_fn=tf.nn.relu, is_training=is_train,
                scope='fc_1', reuse=False)
            map_V = modules.fc_layer(
                map_V, MAP_DIM, use_bias=False, use_bn=True,
                activation_fn=tf.nn.relu, is_training=is_train,
                scope='fc_2', reuse=False)
            map_V = modules.fc_layer(
                map_V, MAP_DIM, use_bias=False, use_bn=False,
                activation_fn=None, is_training=is_train,
                scope='Linear', reuse=False)

        with tf.variable_scope('Classifier') as scope:
            log.warning(scope.name)
            tiled_feat_V = tf.tile(tf.expand_dims(feat_V, axis=1),
                                  [1, self.object_num_k, 1])
            bias = tf.get_variable(name='bias', shape=(),
                                   initializer=tf.zeros_initializer())
            logits = tf.reduce_sum(tiled_feat_V * map_V, axis=-1) + bias

        with tf.name_scope('Loss'):
            labels = self.batches['object']['ground_truth']
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(labels), logits=logits)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('Accuracy'):
            label_token = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
            logit_token = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
            acc = tf.reduce_mean(tf.to_float(
                tf.equal(label_token, logit_token)))
            _, top_k_pred = tf.nn.top_k(logits, k=TOP_K)
            k_label_token = tf.tile(
                tf.expand_dims(label_token, axis=1), [1, TOP_K])
            top_k_acc = tf.reduce_mean(tf.to_float(tf.reduce_any(
                tf.equal(k_label_token, top_k_pred), axis=1)))

        with tf.name_scope('Summary'):
            image = self.batches['object']['image'] / 255.0

            # Visualize prediction as image
            def visualize_prediction(logit, label):
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
            pred_image = visualize_prediction(logits, labels)

            # Visualize prediction as texts
            batch_range = tf.expand_dims(
                tf.range(0, tf.shape(label_token)[0], delta=1), axis=1)
            range_label_token = tf.concat(
                [batch_range, tf.expand_dims(label_token, axis=1)], axis=1)
            label_name = tf.gather_nd(
                self.batches['object']['objects_name'], range_label_token)
            top_k_preds = tf.split(axis=-1, num_or_size_splits=TOP_K,
                                   value=top_k_pred)
            pred_names = []
            for i in range(TOP_K):
                range_top_k_pred = tf.concat(
                    [batch_range, top_k_preds[i]], axis=1)
                pred_names.append(tf.gather_nd(
                    self.batches['object']['objects_name'], range_top_k_pred))
            string_list = ['gt: ', label_name]
            for i in range(TOP_K):
                string_list.extend([', pred({}): '.format(i), pred_names[i]])
            pred_string = tf.string_join(string_list)

        tf.summary.scalar('train/loss', self.loss, collections=['train'])
        tf.summary.scalar('val/loss', self.loss, collections=['val'])

        tf.summary.scalar('train/accuracy', acc, collections=['train'])
        tf.summary.scalar('val/accuracy', acc, collections=['val'])

        tf.summary.scalar('train/top_{}_acc'.format(TOP_K),
                          top_k_acc, collections=['train'])
        tf.summary.scalar('val/top_{}_acc'.format(TOP_K),
                          top_k_acc, collections=['val'])

        tf.summary.image('train_image', image, collections=['train'])
        tf.summary.image('train_prediction_image', pred_image,
                         collections=['train'])
        tf.summary.image('val_image', image, collections=['val'])
        tf.summary.image('val_prediction_image', pred_image,
                         collections=['val'])
        tf.summary.text('train_pred_string', pred_string, collections=['train'])
        tf.summary.text('val_pred_string', pred_string, collections=['val'])

        self.report = {
            'loss': self.loss,
            'accuracy': acc,
            'top_{}_acc'.format(TOP_K): top_k_acc
        }
        self.output = {
            'image': image,
            'prediction_string': pred_string
        }
