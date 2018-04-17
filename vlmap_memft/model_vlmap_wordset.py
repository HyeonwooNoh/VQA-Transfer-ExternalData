import cPickle
import os
import tensorflow as tf

from vlmap import modules

TOP_K = 5
W_DIM = 300  # Word dimension
L_DIM = 1024  # Language dimension
V_DIM = 1024


class Model(object):

    def __init__(self, batch, config, is_train=True):
        self.batch = batch
        self.config = config
        self.data_cfg = config.data_cfg
        self.data_dir = config.data_dir
        self.is_train = is_train

        self.losses = {}
        self.report = {}
        self.mid_result = {}
        self.vis_image = {}

        vocab_path = os.path.join(self.data_dir, 'vocab.pkl')
        self.vocab = cPickle.load(open(vocab_path, 'rb'))

        answer_dict_path = os.path.join(self.data_dir, 'answer_dict.pkl')
        self.answer_dict = cPickle.load(open(answer_dict_path, 'rb'))
        self.num_answer = len(self.answer_dict['vocab'])

        ws_dict_path = os.path.join(self.data_dir, 'wordset_dict5.pkl')
        self.ws_dict = cPickle.load(open(ws_dict_path, 'rb'))
        self.num_ws = len(self.ws_dict['vocab'])

        self.wordset_map = modules.learn_embedding_map(
            self.ws_dict, scope='wordset_map')
        self.v_word_map = modules.LearnGloVe(self.vocab, scope='V_GloVe')
        self.l_word_map = modules.LearnGloVe(self.vocab, scope='L_GloVe')
        self.l_answer_word_map = modules.LearnAnswerGloVe(self.answer_dict)

        self.build()

    def filter_train_vars(self, trainable_vars):
        train_vars = []
        for var in trainable_vars:
            if var.name.split('/')[0] == 'LearnAnswerGloVe':
                train_vars.append(var)
            else: train_vars.append(var)
        return train_vars

    def build(self):
        """
        build network architecture and loss
        """
        self.build_object_predict()
        # self.build_attribute_predict()

        self.mid_result['v_linear_v'] = modules.fc_layer(
            self.batch['image_ft'], V_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='v_linear_v')
        self.build_object_attention()
        self.build_attribute_attention()

        self.build_object_blank_fill()
        self.build_attribute_blank_fill()
        self.build_caption_attention()

        self.loss = 0
        for loss in self.losses.values():
            self.loss = self.loss + loss
        self.report['total_loss'] = self.loss

        # scalar summary
        for key, val in self.report.items():
            tf.summary.scalar('train/{}'.format(key), val,
                              collections=['heavy_train', 'train'])
            tf.summary.scalar('val/{}'.format(key), val,
                              collections=['heavy_val', 'val'])
        # image summary
        for key, val in self.vis_image.items():
            tf.summary.image('train-{}'.format(key), val, max_outputs=10,
                             collections=['heavy_train'])
            tf.summary.image('val-{}'.format(key), val, max_outputs=10,
                             collections=['heavy_val'])

        return self.loss

    def build_object_predict(self):
        """
        object_predict
        """
        # [#obj, #proposal] x [#proposal x feat_dim] -> [#obj,feat_dim]
        V_ft = tf.matmul(self.batch['obj_pred/weights'],
                         self.batch['image_ft'])
        v_linear_l = modules.fc_layer(
            V_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='pooled_linear_l')

        wordset_embed = tf.tanh(tf.nn.embedding_lookup(
            self.wordset_map, self.batch['obj_pred/wordsets']))
        wordset_ft = modules.fc_layer(
            wordset_embed, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.tanh, is_training=self.is_train, scope='wordset_ft2')

        l_linear_l = modules.fc_layer(
            wordset_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_l')

        joint = modules.fc_layer(
            v_linear_l * l_linear_l, L_DIM * 2,
            use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train, scope='joint_fc')
        joint = tf.nn.dropout(joint, 0.5)

        logit = modules.fc_layer(
            joint, self.num_answer,
            use_bias=True, use_bn=False, use_ln=False,
            activation_fn=None, is_training=self.is_train, scope='classifier')
        self.mid_result['obj_pred/logit'] = logit  # [bs, #obj, #answer]

        with tf.name_scope('loss/object_predict'):
            onehot_gt = tf.one_hot(self.batch['obj_pred/labels'],
                                   depth=self.num_answer)
            num_valid_entry = self.batch['obj_pred/num']
            valid_mask = tf.sequence_mask(
                num_valid_entry, maxlen=self.data_cfg.n_obj_pred,
                dtype=tf.float32)
            loss, acc, top_k_acc = \
                self.n_way_classification_loss(logit, onehot_gt, valid_mask)
            self.losses['object_pred'] = loss
            self.report['object_pred_loss'] = loss
            self.report['object_pred_acc'] = acc
            self.report['object_pred_top_{}_acc'.format(TOP_K)] = top_k_acc

    def build_attribute_predict(self):
        """
        attribute_predict
        """
        # [#attr, #proposal] x [#proposal x feat_dim] -> [#attr,feat_dim]
        V_ft = tf.matmul(self.batch['attr_pred/weights'],
                         self.batch['image_ft'])
        v_linear_l = modules.fc_layer(
            V_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='pooled_linear_l')

        L_ft = tf.nn.embedding_lookup(self.l_answer_word_map,
                                      self.batch['attr_pred/object_labels'])
        reg_l_ft = modules.fc_layer(
            L_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.tanh, is_training=self.is_train,
            scope='attr_pred/encode_object_labels')
        self.mid_result['attr_pred/reg_l_ft'] = reg_l_ft

        wordset_embed = tf.tanh(tf.nn.embedding_lookup(
            self.wordset_map, self.batch['attr_pred/random_wordsets']))
        wordset_ft = modules.fc_layer(
            wordset_embed, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.tanh, is_training=self.is_train, scope='wordset_ft_attr')
        context_ft = modules.fc_layer(
            reg_l_ft * wordset_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.tanh, is_training=self.is_train, scope='context_ft_attr')

        l_linear_l = modules.fc_layer(
            context_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_l')

        joint = modules.fc_layer(
            v_linear_l * l_linear_l, L_DIM * 2,
            use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train, scope='joint_fc')
        joint = tf.nn.dropout(joint, 0.5)

        logit = modules.fc_layer(
            joint, self.num_answer,
            use_bias=True, use_bn=False, use_ln=False,
            activation_fn=None, is_training=self.is_train, scope='classifier')
        self.mid_result['attr_pred/logit'] = logit  # [bs, #attr, #answer]

        with tf.name_scope('loss/attr_predict'):
            multilabel_gt = self.batch['attr_pred/random_wordset_labels']
            num_valid_entry = self.batch['attr_pred/num']
            valid_mask = tf.sequence_mask(
                num_valid_entry, maxlen=self.data_cfg.n_attr_pred,
                dtype=tf.float32)
            loss, acc, recall, precision, top_1_prec, top_k_recall = \
                self.binary_classification_loss(logit, multilabel_gt, valid_mask,
                                                depth=self.num_answer)
            self.losses['attr_pred'] = loss
            self.report['attr_pred_loss'] = loss
            self.report['attr_pred_acc'] = acc
            self.report['attr_pred_recall'] = recall
            self.report['attr_pred_precision'] = precision
            self.report['attr_pred_top_1_prec'] = top_1_prec
            self.report['attr_pred_top_{}_recall'.format(TOP_K)] = top_k_recall

    def build_object_attention(self):
        """
        object_attention
        """
        num_V_ft = self.batch['num_boxes']
        v_linear_v = self.mid_result['v_linear_v']

        w_embed = tf.nn.embedding_lookup(self.v_word_map,
                                         self.batch['obj_att/word_tokens'])
        w_L_ft = modules.fc_layer(  # [bs, #proposal, len, L_DIM]
            w_embed, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='v_word_fc')
        w_len = self.batch['obj_att/word_tokens_len']
        mask = tf.sequence_mask(  # [bs, #proposal, len]
            w_len, maxlen=tf.shape(w_L_ft)[-2],
            dtype=tf.float32)
        pooled_w_L_ft = tf.reduce_sum(w_L_ft * tf.expand_dims(mask, axis=-1),
                                      axis=-2)
        pooled_w_L_ft = pooled_w_L_ft / \
            tf.expand_dims(tf.to_float(w_len), axis=-1)

        l_linear_v = modules.fc_layer(
            pooled_w_L_ft, V_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_v')

        tile_v_linear_v = tf.tile(tf.expand_dims(v_linear_v, axis=1),
                                  [1, self.data_cfg.n_obj_att, 1, 1])
        flat_tile_v_linear_v = tf.reshape(tile_v_linear_v,
                                          [-1, self.data_cfg.max_box_num, V_DIM])
        tile_num_V_ft = tf.tile(tf.expand_dims(num_V_ft, axis=1),
                                [1, self.data_cfg.n_obj_att])
        flat_tile_num_V_ft = tf.reshape(tile_num_V_ft, [-1])

        flat_l_linear_v = tf.reshape(l_linear_v, [-1, V_DIM])

        # flat_att_logit: [bs * #obj, num_proposal]
        flat_att_logit = modules.hadamard_attention(
            flat_tile_v_linear_v, flat_tile_num_V_ft, flat_l_linear_v,
            use_ln=False, is_train=self.is_train, normalizer=None)

        n_entry = self.data_cfg.n_obj_att
        n_proposal = self.data_cfg.max_box_num
        logit = tf.reshape(flat_att_logit, [-1, n_entry, n_proposal])

        with tf.name_scope('loss/object_attend'):
            multilabel_gt = tf.to_float(
                tf.greater(self.batch['obj_att/att_scores'], 0.5))
            num_valid_entry = self.batch['obj_att/num']
            valid_mask = tf.sequence_mask(
                num_valid_entry, maxlen=self.data_cfg.n_obj_att,
                dtype=tf.float32)
            loss, acc, recall, precision, top_1_prec, top_k_recall = \
                self.binary_classification_loss(logit, multilabel_gt, valid_mask,
                                                depth=self.data_cfg.max_box_num)
            self.losses['object_att'] = loss
            self.report['object_att_loss'] = loss
            self.report['object_att_acc'] = acc
            self.report['object_att_recall'] = recall
            self.report['object_att_precision'] = precision
            self.report['object_att_top_1_prec'] = top_1_prec
            self.report['object_att_top_{}_recall'.format(TOP_K)] = top_k_recall

    def build_attribute_attention(self):
        """
        attribute_attention
        """
        num_V_ft = self.batch['num_boxes']
        v_linear_v = self.mid_result['v_linear_v']

        w_embed = tf.nn.embedding_lookup(self.v_word_map,
                                         self.batch['attr_att/word_tokens'])
        w_L_ft = modules.fc_layer(
            w_embed, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='v_word_fc')

        w_len = self.batch['attr_att/word_tokens_len']
        mask = tf.sequence_mask(  # [bs, #proposal, len]
            w_len, maxlen=tf.shape(w_L_ft)[-2],
            dtype=tf.float32)
        pooled_w_L_ft = tf.reduce_sum(w_L_ft * tf.expand_dims(mask, axis=-1),
                                      axis=-2)
        pooled_w_L_ft = pooled_w_L_ft / \
            tf.expand_dims(tf.to_float(w_len), axis=-1)

        l_linear_v = modules.fc_layer(
            pooled_w_L_ft, V_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_v')

        tile_v_linear_v = tf.tile(tf.expand_dims(v_linear_v, axis=1),
                                  [1, self.data_cfg.n_attr_att, 1, 1])
        flat_tile_v_linear_v = tf.reshape(tile_v_linear_v,
                                          [-1, self.data_cfg.max_box_num, V_DIM])
        tile_num_V_ft = tf.tile(tf.expand_dims(num_V_ft, axis=1),
                                [1, self.data_cfg.n_attr_att])
        flat_tile_num_V_ft = tf.reshape(tile_num_V_ft, [-1])

        flat_l_linear_v = tf.reshape(l_linear_v, [-1, V_DIM])

        # flat_att_logit: [bs * #attr, num_proposal]
        flat_att_logit = modules.hadamard_attention(
            flat_tile_v_linear_v, flat_tile_num_V_ft, flat_l_linear_v,
            use_ln=False, is_train=self.is_train, normalizer=None)

        n_entry = self.data_cfg.n_attr_att
        n_proposal = self.data_cfg.max_box_num
        logit = tf.reshape(flat_att_logit, [-1, n_entry, n_proposal])

        with tf.name_scope('loss/attr_attend'):
            multilabel_gt = tf.to_float(
                tf.greater(self.batch['attr_att/att_scores'], 0.5))
            num_valid_entry = self.batch['attr_att/num']
            valid_mask = tf.sequence_mask(
                num_valid_entry, maxlen=self.data_cfg.n_attr_att,
                dtype=tf.float32)
            loss, acc, recall, precision, top_1_prec, top_k_recall = \
                self.binary_classification_loss(logit, multilabel_gt, valid_mask,
                                                depth=self.data_cfg.max_box_num)
            self.losses['attr_att'] = loss
            self.report['attr_att_loss'] = loss
            self.report['attr_att_acc'] = acc
            self.report['attr_att_recall'] = recall
            self.report['attr_att_precision'] = precision
            self.report['attr_att_top_1_prec'] = top_1_prec
            self.report['attr_att_top_{}_recall'.format(TOP_K)] = top_k_recall

    def build_object_blank_fill(self):
        """
        object_blank_fill
        """
        # [#obj, #proposal] x [#proposal x feat_dim] -> [#obj,feat_dim]
        V_ft = tf.matmul(self.batch['obj_blank_fill/weights'],
                         self.batch['image_ft'])
        v_linear_l = modules.fc_layer(
            V_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='pooled_linear_l')

        blank_embed = tf.nn.embedding_lookup(  # [bs, #proposal, len, W_DIM]
            self.l_word_map, self.batch['obj_blank_fill/blanks'])
        blank_len = self.batch['obj_blank_fill/blanks_len']
        blank_maxlen = tf.shape(blank_embed)[-2]
        flat_blank_ft = modules.encode_L(  # [bs * #proposal, L_DIM]
            tf.reshape(blank_embed, [-1, blank_maxlen, W_DIM]),
            tf.reshape(blank_len, [-1]), L_DIM,
            scope='encode_L_blank', cell_type='GRU')
        blank_ft = tf.reshape(
            flat_blank_ft, [-1, self.data_cfg.n_obj_bf, L_DIM])

        wordset_embed = tf.tanh(tf.nn.embedding_lookup(
            self.wordset_map, self.batch['obj_blank_fill/wordsets']))
        wordset_ft = modules.fc_layer(
            wordset_embed, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.tanh, is_training=self.is_train, scope='wordset_ft')
        context_ft = modules.fc_layer(
            blank_ft * wordset_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.tanh, is_training=self.is_train, scope='context_ft')

        l_linear_l = modules.fc_layer(
            context_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_l')

        joint = modules.fc_layer(
            v_linear_l * l_linear_l, L_DIM * 2,
            use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train, scope='joint_fc')
        joint = tf.nn.dropout(joint, 0.5)

        logit = modules.fc_layer(
            joint, self.num_answer,
            use_bias=True, use_bn=False, use_ln=False,
            activation_fn=None, is_training=self.is_train, scope='classifier')
        self.mid_result['obj_blank_fill/logit'] = logit  # [bs, #obj, #answer]

        with tf.name_scope('loss/obj_blank_fill'):
            onehot_gt = tf.one_hot(self.batch['obj_blank_fill/fills'],
                                   depth=self.num_answer)
            num_valid_entry = self.batch['obj_blank_fill/num']
            valid_mask = tf.sequence_mask(
                num_valid_entry, maxlen=self.data_cfg.n_obj_bf,
                dtype=tf.float32)
            loss, acc, top_k_acc = \
                self.n_way_classification_loss(logit, onehot_gt, valid_mask)
            self.losses['obj_blank_fill'] = loss
            self.report['obj_blank_fill_loss'] = loss
            self.report['obj_blank_fill_acc'] = acc
            self.report['obj_blank_fill_top_{}_acc'.format(TOP_K)] = top_k_acc

    def build_attribute_blank_fill(self):
        """
        attribute_blank_fill
        """
        # [#obj, #proposal] x [#proposal x feat_dim] -> [#obj,feat_dim]
        V_ft = tf.matmul(self.batch['attr_blank_fill/weights'],
                         self.batch['image_ft'])
        v_linear_l = modules.fc_layer(
            V_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='pooled_linear_l')

        blank_embed = tf.nn.embedding_lookup(  # [bs, #proposal, len, W_DIM]
            self.l_word_map, self.batch['attr_blank_fill/blanks'])
        blank_len = self.batch['attr_blank_fill/blanks_len']
        blank_maxlen = tf.shape(blank_embed)[-2]
        flat_blank_ft = modules.encode_L(  # [bs * #proposal, L_DIM]
            tf.reshape(blank_embed, [-1, blank_maxlen, W_DIM]),
            tf.reshape(blank_len, [-1]), L_DIM,
            scope='encode_L_blank', cell_type='GRU')
        blank_ft = tf.reshape(
            flat_blank_ft, [-1, self.data_cfg.n_obj_bf, L_DIM])

        wordset_embed = tf.tanh(tf.nn.embedding_lookup(
            self.wordset_map, self.batch['attr_blank_fill/wordsets']))
        wordset_ft = modules.fc_layer(
            wordset_embed, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.tanh, is_training=self.is_train, scope='wordset_ft')
        context_ft = modules.fc_layer(
            blank_ft * wordset_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.tanh, is_training=self.is_train, scope='context_ft')

        l_linear_l = modules.fc_layer(
            context_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_l')

        joint = modules.fc_layer(
            v_linear_l * l_linear_l, L_DIM * 2,
            use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train, scope='joint_fc')
        joint = tf.nn.dropout(joint, 0.5)

        logit = modules.fc_layer(
            joint, self.num_answer,
            use_bias=True, use_bn=False, use_ln=False,
            activation_fn=None, is_training=self.is_train, scope='classifier')
        self.mid_result['attr_blank_fill/logit'] = logit  # [bs, #attr, #answer]

        with tf.name_scope('loss/attr_blank_fill'):
            onehot_gt = tf.one_hot(self.batch['attr_blank_fill/fills'],
                                   depth=self.num_answer)
            num_valid_entry = self.batch['attr_blank_fill/num']
            valid_mask = tf.sequence_mask(
                num_valid_entry, maxlen=self.data_cfg.n_attr_bf,
                dtype=tf.float32)
            loss, acc, top_k_acc = \
                self.n_way_classification_loss(logit, onehot_gt, valid_mask)
            self.losses['attr_blank_fill'] = loss
            self.report['attr_blank_fill_loss'] = loss
            self.report['attr_blank_fill_acc'] = acc
            self.report['attr_blank_fill_top_{}_acc'.format(TOP_K)] = top_k_acc

    def build_caption_attention(self):
        """
        caption_attention
        """
        num_V_ft = self.batch['num_boxes']
        v_linear_v = self.mid_result['v_linear_v']

        w_embed = tf.nn.embedding_lookup(self.v_word_map,
                                         self.batch['cap_att/word_tokens'])
        w_L_ft = modules.fc_layer(  # [bs, #proposal, len, L_DIM]
            w_embed, L_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='v_word_fc')
        w_len = self.batch['cap_att/word_tokens_len']
        mask = tf.sequence_mask(  # [bs, #proposal, len]
            w_len, maxlen=tf.shape(w_L_ft)[-2],
            dtype=tf.float32)
        pooled_w_L_ft = tf.reduce_sum(w_L_ft * tf.expand_dims(mask, axis=-1),
                                      axis=-2)
        pooled_w_L_ft = pooled_w_L_ft / \
            tf.expand_dims(tf.to_float(w_len), axis=-1)

        l_linear_v = modules.fc_layer(
            pooled_w_L_ft, V_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_v')

        tile_v_linear_v = tf.tile(tf.expand_dims(v_linear_v, axis=1),
                                  [1, self.data_cfg.n_cap_att, 1, 1])
        flat_tile_v_linear_v = tf.reshape(tile_v_linear_v,
                                          [-1, self.data_cfg.max_box_num, V_DIM])
        tile_num_V_ft = tf.tile(tf.expand_dims(num_V_ft, axis=1),
                                [1, self.data_cfg.n_cap_att])
        flat_tile_num_V_ft = tf.reshape(tile_num_V_ft, [-1])

        flat_l_linear_v = tf.reshape(l_linear_v, [-1, V_DIM])

        # flat_att_logit: [bs * #obj, num_proposal]
        flat_att_logit = modules.hadamard_attention(
            flat_tile_v_linear_v, flat_tile_num_V_ft, flat_l_linear_v,
            use_ln=False, is_train=self.is_train, normalizer=None)

        n_entry = self.data_cfg.n_cap_att
        n_proposal = self.data_cfg.max_box_num
        logit = tf.reshape(flat_att_logit, [-1, n_entry, n_proposal])

        with tf.name_scope('loss/caption_attend'):
            multilabel_gt = tf.to_float(
                tf.greater(self.batch['cap_att/att_scores'], 0.5))
            num_valid_entry = self.batch['cap_att/num']
            valid_mask = tf.sequence_mask(
                num_valid_entry, maxlen=self.data_cfg.n_cap_att,
                dtype=tf.float32)
            loss, acc, recall, precision, top_1_prec, top_k_recall = \
                self.binary_classification_loss(logit, multilabel_gt, valid_mask,
                                                depth=self.data_cfg.max_box_num)
            self.losses['caption_att'] = loss
            self.report['caption_att_loss'] = loss
            self.report['caption_att_acc'] = acc
            self.report['caption_att_recall'] = recall
            self.report['caption_att_precision'] = precision
            self.report['caption_att_top_1_prec'] = top_1_prec
            self.report['caption_att_top_{}_recall'.format(TOP_K)] = top_k_recall

    def n_way_classification_loss(self, logits, labels, mask=None):
        # Loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits, dim=-1)
        if mask is None:
            loss = tf.reduce_mean(cross_entropy)
        else:
            loss = tf.reduce_sum(cross_entropy * mask) / tf.reduce_sum(mask)

        # Top-1 Accuracy
        label_token = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        logit_token = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        correct = tf.to_float(tf.equal(label_token, logit_token))
        if mask is None:
            acc = tf.reduce_mean(correct)
        else:
            acc = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)

        # Top-K Accuracy
        _, top_k_pred = tf.nn.top_k(logits, k=TOP_K)
        k_label_token = tf.tile(
            tf.expand_dims(label_token, axis=-1), [1, 1, TOP_K])

        top_k_correct = tf.to_float(tf.reduce_any(
            tf.equal(k_label_token, top_k_pred), axis=-1))
        if mask is None:
            top_k_acc = tf.reduce_mean(top_k_correct)
        else:
            top_k_acc = tf.reduce_sum(top_k_correct * mask) / tf.reduce_sum(mask)

        return loss, acc, top_k_acc

    def binary_classification_loss(self, logits, labels, mask=None, depth=None):
        # Loss
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=-1)  # sum over dim
        if mask is None:
            loss = tf.reduce_mean(cross_entropy)
        else:
            loss = tf.reduce_sum(cross_entropy * mask) / tf.reduce_sum(mask)

        if depth is not None:
            pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
            onehot_pred = tf.one_hot(pred, depth=depth, dtype=tf.float32)
            top_1_prec = tf.reduce_sum(onehot_pred * labels, axis=-1)
            if mask is None:
                top_1_prec = tf.reduce_mean(top_1_prec)
            else:
                top_1_prec = tf.reduce_sum(top_1_prec * mask) / \
                    tf.reduce_sum(mask)

            _, top_k_pred = tf.nn.top_k(logits, k=TOP_K)  # [bs, n, TOP_K]
            onehot_top_k_pred = tf.one_hot(  # [bs, n, TOP_K, depth]
                top_k_pred, depth=depth, dtype=tf.float32)
            top_k_hot_pred = tf.reduce_sum(
                onehot_top_k_pred, axis=2)  # [bs, n, depth] n-hot vector

            num_correct_1 = tf.reduce_sum(top_k_hot_pred * labels, axis=-1)
            num_label_1 = tf.minimum(
                tf.reduce_sum(labels, axis=-1), TOP_K)
            top_k_recall = num_correct_1 / num_label_1
            if mask is None:
                top_k_recall = tf.reduce_mean(top_k_recall)
            else:
                top_k_recall = tf.reduce_sum(top_k_recall * mask) / \
                    tf.reduce_sum(mask)

        binary_pred = tf.cast(tf.greater(logits, 0), tf.int32)
        binary_label = tf.cast(labels, tf.int32)

        tp = tf.to_float(tf.logical_and(tf.equal(binary_pred, 1),
                                        tf.equal(binary_label, 1)))
        fp = tf.to_float(tf.logical_and(tf.equal(binary_pred, 1),
                                        tf.equal(binary_label, 0)))
        tn = tf.to_float(tf.logical_and(tf.equal(binary_pred, 0),
                                        tf.equal(binary_label, 0)))
        fn = tf.to_float(tf.logical_and(tf.equal(binary_pred, 0),
                                        tf.equal(binary_label, 1)))

        if mask is not None:
            expand_mask = tf.expand_dims(mask, axis=-1)
            tp = tp * expand_mask
            fp = fp * expand_mask
            tn = tn * expand_mask
            fn = fn * expand_mask

        n_tp = tf.reduce_sum(tp)  # true positive
        n_fp = tf.reduce_sum(fp)  # false positive
        n_tn = tf.reduce_sum(tn)  # true negative
        n_fn = tf.reduce_sum(fn)  # false negative

        acc = (n_tp + n_tn) / (n_tp + n_fp + n_tn + n_fn + 1e-12)
        recall = (n_tp) / (n_tp + n_fn + 1e-12)
        precision = (n_tp) / (n_tp + n_fp + 1e-12)

        ret = [loss, acc, recall, precision]

        if depth is not None:
            ret.extend([top_1_prec, top_k_recall])
        return tuple(ret)
