import h5py
import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from util import log
from vlmap import modules

TOP_K = 5
W_DIM = 300  # Word dimension
L_DIM = 512  # Language dimension
MAP_DIM = 512
V_DIM = 512
ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'

ROI_SZ = 5


class Model(object):

    def __init__(self, batch, config, is_train=True):
        self.batch = batch
        self.config = config
        self.data_cfg = config.dataset_config

        self.losses = {}
        self.report = {}
        self.output = {}

        self.vocab = json.load(open(config.vocab_path, 'r'))
        self.glove_map = modules.GloVe(config.glove_path)

        # model parameters
        self.ft_enc_I = config.ft_enc_I
        self.no_V_grad_enc_L = config.no_V_grad_enc_L
        self.use_relation = config.use_relation
        self.description_task = config.description_task
        self.decoder_type = config.decoder_type
        self.num_aug_retrieval = config.num_aug_retrieval




        self.no_object = config.no_object
        self.no_region = config.no_region
        self.use_blank_fill = config.use_blank_fill

        self.object_batch_size = config.object_batch_size
        self.region_batch_size = config.region_batch_size

        self.object_num_k = config.object_num_k
        self.object_max_name_len = config.object_max_name_len

        self.region_max_len = config.region_max_len

        # model parameters
        self.finetune_enc_I = config.finetune_enc_I
        self.no_V_grad_enc_L = config.no_V_grad_enc_L
        self.no_V_grad_dec_L = config.no_V_grad_dec_L
        self.no_L_grad_dec_L = config.no_L_grad_dec_L
        self.use_embed_transform = config.use_embed_transform
        self.use_dense_predictor = config.use_dense_predictor
        self.no_glove = config.no_glove

        self.wordset = modules.used_wordset(config.used_wordset_path)
        self.wordset_vocab = {}
        with h5py.File(config.used_wordset_path, 'r') as f:
            wordset = list(f['used_wordset'].value)
            self.wordset_vocab['vocab'] = [self.vocab['vocab'][w]
                                           for w in wordset]
            self.wordset_vocab['dict'] = {w: i for i, w in
                                          enumerate(self.wordset_vocab['vocab'])}

        if self.no_glove:
            self.glove_all = modules.learn_embedding_map(self.vocab)
        else: self.glove_all = modules.glove_embedding_map(self.vocab)
        self.glove_wordset = tf.nn.embedding_lookup(self.glove_all,
                                                    self.wordset)
        predictor_embed = self.glove_wordset
        if self.use_embed_transform:
            predictor_embed = modules.embedding_transform(
                predictor_embed, W_DIM, L_DIM, is_train=is_train)
        if self.use_dense_predictor:
            self.word_predictor = tf.layers.Dense(
                len(self.wordset_vocab['vocab']), use_bias=True, name='WordPredictor')
        else:
            self.word_predictor = modules.WordPredictor(predictor_embed,
                                                        trainable=is_train,
                                                        name='WordPredictor')

        self.build(is_train=is_train)

    def filter_vars(self, all_vars):
        enc_I_vars = []
        learn_v_vars = []  # variables learning from vision loss
        learn_l_vars = []  # variables learning from language loss
        for var in all_vars:
            if var.name.split('/')[0] == 'resnet_v1_50':
                enc_I_vars.append(var)
                if self.finetune_enc_I:
                    learn_v_vars.append(var)
            elif var.name.split('/')[0] == 'language_encoder':
                learn_l_vars.append(var)
                if not self.no_V_grad_enc_L:
                    learn_v_vars.append(var)
            elif var.name.split('/')[0] == 'language_decoder':
                if not self.no_L_grad_dec_L:
                    learn_l_vars.append(var)
                if not self.no_V_grad_dec_L:
                    learn_v_vars.append(var)
            else:
                learn_v_vars.append(var)
        return enc_I_vars, learn_v_vars, learn_l_vars

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


    def description_loss(self, logits, gt_seq, mask):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=gt_seq, logits=logits)
        loss = tf.reduce_sum(cross_entropy * mask) / tf.reduce_sum(mask)


    def n_way_classification_loss(self, logits, labels, mask=None):
        # Loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_V2(
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
            tf.expand_dims(label_token, axis=1), [1, TOP_K])

        top_k_correct = tf.to_float(tf.reduce_any(
            tf.equal(k_label_token, top_k_pred), axis=1))
        if mask is None:
            top_k_acc = tf.reduce_mean(top_k_correct)
        else:
            top_k_acc = tf.reduce_sum(top_k_correct * mask) / tf.reduce_sum(mask)

        return loss, acc, top_k_acc


    def binary_classification_loss(self, logits, labels, mask=None):
        # Loss
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits, dim=-1)
        cross_entropy = tf.reduce_mean(cross_entropy, axis=-1)
        if mask is None:
            loss = tf.reduce_mean(cross_entropy)
        else:
            loss = tf.reduce_sum(cross_entropy * mask) / tf.reduce_sum(mask)

        binary_pred = tf.cast(logits > 0, tf.int32)
        binary_label = tf.cast(labels, tf.int32)

        tp = tf.to_float(tf.logical_and(binary_pred == 1, binary_label == 1))
        fp = tf.to_float(tf.logical_and(binary_pred == 1, binary_label == 0))
        tn = tf.to_float(tf.logical_and(binary_pred == 0, binary_label == 0))
        fn = tf.to_float(tf.logical_and(binary_pred == 0, binary_label == 1))

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

        acc = (n_tp + n_tn) / (n_tp + n_fp + n_tn + n_fn)
        recall = (n_tp) / (n_tp + n_fn)
        precision = (n_tp) / (n_tp + n_fp)

        return loss, acc, recall, precision


    def flat_description_loss(self, logits_flat,
                              desc_flat, desc_len_flat, used_mask):
        desc_maxlen = tf.shape(desc_flat)[-1]
        used_mask_flat = tf.expand_dims(tf.reshape(used_mask, [-1]), axis=-1)
        mask = tf.sequence_mask(
            desc_len_flat, maxlen=desc_maxlen, dtype=tf.float32) * used_mask_flat

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=desc_flat, logits=logits_flat)
        loss = tf.reduce_sum(cross_entropy * mask) / \
            tf.reduce_sum(mask)
        return loss


    def flat_description_accuracy(self, pred_flat, pred_len_flat,
                                  desc_flat, desc_len_flat, used_mask):
        max_len = tf.reduce_max(tf.concat(
            [pred_len_flat, desc_len_flat], axis=0))
        p_sz = tf.shape(pred_flat)
        d_sz = tf.shape(desc_flat)
        max_len = tf.maximum(max_len, p_sz[1])
        max_len = tf.maximum(max_len, d_sz[1])
        # Dynamic padding
        p_pad = tf.zeros([p_sz[0], max_len - p_sz[1]], dtype=pred_flat.dtype)
        d_pad = tf.zeros([d_sz[0], max_len - d_sz[1]], dtype=desc_flat.dtype)
        pred_flat = tf.concat([pred_flat, p_pad], axis=1)
        desc_flat = tf.concat([desc_flat, d_pad], axis=1)
        # Mask construction
        used_mask_flat = tf.expand_dims(tf.reshape(used_mask, [-1]), axis=-1)
        mask = tf.sequence_mask(
            desc_len_flat, maxlen=max_len,
            dtype=tf.float32, name='mask') * used_mask_flat
        min_mask = tf.sequence_mask(
            tf.minimum(pred_len_flat, desc_len_flat), maxlen=max_len,
            dtype=tf.float32, name='min_mask') * used_mask_flat
        max_mask = tf.sequence_mask(
            tf.maximum(pred_len_flat, desc_len_flat), maxlen=max_len,
            dtype=tf.float32, name='max_mask') * used_mask_flat
        # Accuracy
        token_acc = tf.reduce_sum(
            tf.to_float(tf.equal(pred_flat, desc_flat)) * min_mask) / \
            tf.reduce_sum(max_mask)
        seq_acc = tf.logical_and(
            tf.reduce_all(tf.equal(pred_flat * mask_int,
                                   desc_flat * mask_int), axis=-1),
            tf.equal(desc_len_flat, pred_len_flat))
        seq_acc = tf.reduce_sum(tf.to_float(seq_acc) * used_mask_flat) / \
            tf.reduce_sum(used_mask_flat)
        return token_acc, seq_acc


    def sequence_accuracy(self, pred, pred_len, seq, seq_len, mask):
        max_len = tf.reduce_max(tf.concat([pred_len, seq_len], axis=0))
        p_sz = tf.shape(pred)
        s_sz = tf.shape(seq)
        max_len = tf.maximum(max_len, p_sz[1])
        max_len = tf.maximum(max_len, s_sz[1])
        # Dynamic padding
        p_pad = tf.zeros([p_sz[0], max_len - p_sz[1]], dtype=pred.dtype)
        s_pad = tf.zeros([s_sz[0], max_len - s_sz[1]], dtype=seq.dtype)
        pred = tf.concat([pred, p_pad], axis=1)
        seq = tf.concat([seq, s_pad], axis=1)
        # Mask construction
        mask = tf.sequence_mask(seq_len, maxlen=max_len,
                                dtype=tf.float32, name='mask')
        min_mask = tf.sequence_mask(tf.minimum(pred_len, seq_len),
                                    maxlen=max_len,
                                    dtype=tf.float32, name='min_mask')
        max_mask = tf.sequence_mask(tf.maximum(pred_len, seq_len),
                                    maxlen=max_len,
                                    dtype=tf.float32, name='max_mask')
        # Accuracy
        token_acc = tf.reduce_sum(tf.to_float(tf.equal(
            pred, seq)) * min_mask) / tf.reduce_sum(max_mask, axis=[0, 1])
        seq_acc = tf.reduce_mean(tf.to_float(tf.logical_and(
            tf.reduce_all(tf.equal(pred * tf.cast(mask, tf.int32),
                                   seq * tf.cast(mask, tf.int32)), axis=-1),
            tf.equal(seq_len, pred_len))))
        return token_acc, seq_acc

    def seq2string(self, seq, seq_len):
        def seq2string_fn(in_seq, in_seq_len):
            seq_strings = []
            for s, s_len in zip(in_seq, in_seq_len):
                seq_strings.append(
                    ' '.join([self.wordset_vocab['vocab'][i]
                              for i in s[:s_len]]))
            return np.array(seq_strings)
        seq_string = tf.py_func(seq2string_fn, [seq, seq_len], tf.string)
        return seq_string

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


    def aug_retrieval(V, gt, num_aug, num_b, num_k, dim, scope='aug_retrieval'):
        with tf.name_scope(scope):
            sz = tf.shape(V)
            rotate_idx = [tf.mod(tf.range(-1 - i, sz[0] - 1 - i), sz[0])
                          for i in num_aug]
            rotate_idx = tf.stack(rotate_idx, axis=0)
            rotate_V = tf.transpose(tf.reshape(
                tf.gather(V, tf.reshape(rotate_idx, [-1]), axis=0),
                [num_aug, -1, num_b, num_k, dim]), [1, 2, 0, 3, 4])
            rotate_V = tf.reshape(rotate_V, [-1, num_b, num_k * num_aug, dim])
            V = tf.concat([V, rotate_V], axis=2)
            gt_pad = tf.zeros([sz[0], num_b, num_k * new_aug], dtype=tf.float32)
            gt = tf.concat([gt, gt_pad], axis=2)
            return V, gt

    def build(self, is_train=True):
        """
        build network architecture and loss
        """

        """
        Visual features
        """
        # feat_V
        enc_I = modules.encode_I(self.batch['image'],
                                 is_train=self.ft_enc_I)
        if not self.ft_enc_I: enc_I = tf.stop_gradient(enc_I)

        I_lowdim = modules.I_reduce_dim(enc_I, V_DIM, scope='I_reduce_dim',
                                        is_train=is_train)

        roi_ft = modules.roi_pool(I_lowdim, self.batch['normal_box'],
                                  ROI_SZ, ROI_SZ)

        # _flat regards "num box" dimension as a batch
        roi_ft_flat = tf.reshape(roi_ft, [-1, ROI_SZ, ROI_SZ, V_DIM])

        V_ft_flat = modules.I2V(roi_ft_flat, V_DIM, V_DIM, is_train=is_train)
        V_ft = tf.reshape(V_ft_flat, [-1, self.data_cfg.num_box, V_DIM])

        """
        Classification: object, attribute, relationship
        """
        target_entry = ['object', 'attribute']
        if self.use_relation: target_entry.append('relationship')
        for key in target_entry:
            log.info('Language: {}'.format(key))
            num_b = self.data_cfg.num_entry_box[key]
            num_k = self.data_cfg.num_k

            token = self.batch['{}_candidate'.format(key)]
            token_len = self.batch['{}_candidate_len'.format(key)]
            token_maxlen = tf.shape(token)[-1]
            # encode_L
            embed_seq = tf.nn.embedding_lookup(self.glove_map, token)
            enc_L_flat = modules.encode_L(
                tf.reshape(embed_seq, [-1, token_maxlen, W_DIM]),
                tf.reshape(token_len, [-1]), L_DIM)
            if self.no_V_grad_enc_L: enc_L_flat = tf.stop_gradient(enc_L_flat)
            enc_L = tf.reshape(enc_L_flat, [-1, num_b, num_k, L_DIM])

            # L2V mapping
            map_V = modules.L2V(enc_L, MAP_DIM, V_DIM, is_train=is_train)

            # gather target V_ft
            box_idx = modules.batch_box(self.batch['{}_box_idx'],
                                        offset=self.data_cfg.num_box)
            box_V_ft_flat = tf.gather(V_ft_flat, tf.reshape(box_idx, [-1]))
            box_V_ft = tf.reshape(box_V_ft_flat, [-1, num_b, V_DIM])

            # classification
            logits = modules.batch_word_classifier(box_V_ft, map_V)

            with tf.name_scope('{}_classification_loss'.format(key)):
                gt = batch['{}_selection_gt'.format(key)]
                num_used_box = batch['{}_num_used_box'.format(key)]
                used_mask = tf.sequence_mask(num_used_box, dtype=tf.float32)

                if key == 'attribute':
                    loss, acc, recall, precision = \
                        self.binary_classification_loss(logits, gt, used_mask)
                    self.losses[key] = loss
                    self.report['{}_loss'.format(key)] = loss
                    self.report['{}_acc'.format(key)] = acc
                    self.report['{}_recall'.format(key)] = recall
                    self.report['{}_precision'.format(key)] = precision
                else:
                    loss, acc, top_k_acc = \
                        self.n_way_classification_loss(logits, gt, used_mask)
                    self.losses[key] = loss
                    self.report['{}_loss'.format(key)] = loss
                    self.report['{}_acc'.format(key)] = acc
                    self.report['{}_top_k_acc'.format(key)] = top_k_acc

        """
        Region description
        """
        # Select V_ft for descriptions
        num_desc_box = self.data_cfg.num_entry_box['region']
        desc_box_idx = modules.batch_box(self.batch['desc_box_idx'],
                                         offset=self.data_cfg.num_box)
        desc_box_V_ft_flat = tf.gather(V_ft_flat, tf.reshape(desc_box_idx, [-1]))
        desc_box_V_ft = tf.reshape(desc_box_V_ft, [-1, num_desc_box, V_DIM])

        # Metric learning
        desc = self.batch['desc']
        desc_len = self.batch['desc_len']
        desc_maxlen = tf.shape(desc)[-1]

        # encode desc_map_V
        desc_embed_seq = tf.nn.embedding_lookup(self.glove_map, desc)
        desc_L_flat = modules.encode_L(
            tf.reshape(desc_embed_seq, [-1, desc_maxlen, W_DIM]),
            tf.reshape(desc_len, [-1]), L_DIM)
        if self.no_V_grad_enc_L: desc_L_flat = tf.stop_gradient(desc_L_flat)

        desc_map_V_flat = modules.L2V(desc_L_flat, MAP_DIM, V_DIM,
                                      is_train=is_train)
        desc_map_V = tf.reshape(desc_map_V_flat, [-1, num_desc_box, V_DIM])

        # Language retrieval - for each region - classification over descriptions
        lr_num_k = self.data_cfg.lr_num_k
        lr_desc_idx_flat = modules.batch_box(
            tf.reshape(self.batch['lr_desc_idx'], [-1, num_desc_box, lr_num_k]),
            offset=num_desc_box)
        lr_map_V_flat = tf.gather(desc_map_V_flat,
                                  tf.reshape(lr_desc_idx_flat, [-1]))
        lr_map_V = tf.reshape(lr_map_V_flat, [-1, num_desc_box, lr_num_k, V_DIM])
        lr_gt = batch['lr_gt']

        if self.num_aug_retrieval > 0:
            lr_map_V, lr_gt = self.aug_retrieval(
                lr_map_V, lr_gt, self.num_aug_retrieval, num_desc_box,
                lr_num_k, V_DIM, scope='aug_LR')

        # Language Retrieval Classifier
        lr_logits = modules.batch_word_classifier(desc_box_V_ft, lr_map_V)

        with tf.name_scope('LR_classification_loss'):
            num_used_desc = batch['num_used_desc']
            used_desc_mask = tf.sequence_mask(num_used_desc, dtype=tf.float32)

            loss, acc, top_k_acc = self.n_way_classification_loss(
                lr_logits, lr_gt, used_desc_mask)
            self.losses['retrieval_L'] = loss
            self.report['retrieval_L_loss'] = loss
            self.report['retrieval_L_acc'] = acc
            self.report['retrieval_L_top_k_acc'] = top_k_acc

        # Image retrieval - for each description - classification over images
        ir_num_k = self.data_cfg.ir_num_k
        ir_box_idx_flat = modules.batch_box(
            tf.reshape(self.batch['ir_box_idx'], [-1, num_desc_box, ir_num_k]),
            offset=num_desc_box)
        ir_box_V_ft_flat = tf.gather(desc_box_V_ft_flat,
                                     tf.reshape(ir_box_V_ft_flat, [-1]))
        ir_box_V = tf.reshape(ir_box_V_ft_flat,
                              [-1, num_desc_box, ir_num_k, V_DIM])
        ir_gt = batch['ir_gt']

        if self.num_aug_retrieval > 0:
            ir_box_V, ir_gt = self.aug_retrieval(
                ir_box_V, ir_gt, self.num_aug_retrieval, num_desc_box,
                ir_num_k, V_DIM, scope='aug_IR')

        # Image Retrieval Classifier
        ir_logits = modules.batch_word_classifier(desc_map_V_ft, ir_box_V)

        with tf.name_scope('IR_classification_loss'):
            num_used_desc = batch['num_used_desc']
            used_desc_mask = tf.sequence_mask(num_used_desc, dtype=tf.float32)

            loss, acc, top_k_acc = self.n_way_classification_loss(
                ir_logits, ir_gt, used_desc_mask)
            self.losses['retrieval_I'] = loss
            self.report['retrieval_I_loss'] = loss
            self.report['retrieval_I_acc'] = acc
            self.report['retrieval_I_top_k_acc'] = top_k_acc

        # Description / blank-fill task

        # V2L mapping
        desc_box_map_L, V2L_hidden = modules.V2L(
            desc_box_V_ft, MAP_DIM, L_DIM, is_train=is_train)
        in_L = desc_box_map_L  # language feature used for the decoding

        # Add blank-fill feature to mapped language for decoding
        if description_task == 'blank-fill':
            blank_desc = self.batch['blank_desc']
            blank_desc_len = self.batch['blank_desc_len']
            blank_max_len = tf.shape(blank_desc)[-1]

            blank_embed_seq = tf.nn.embedding_lookup(self.glove_map, blank_desc)
            blank_L_flat = modules.encode_L(
                tf.reshape(blank_embed_seq, [-1, blank_max_len, W_DIM]),
                tf.reshape(blank_desc_len, [-1]), L_DIM)
            blank_L = tf.reshape(blank_L_flat, [-1, num_desc_box, L_DIM])
            in_L = in_L + blank_L

        # Decode
        in_L_flat = tf.reshape(in_L, [-1, L_DIM])
        desc_flat = tf.reshape(desc, [-1, desc_maxlen])
        desc_len_flat = tf.reshape(desc_len, [-1])
        if self.decoder_type == 'glove_et':
            pred_embed = modules.embed_transform(
                self.glove_map, L_DIM, L_DIM, is_train=is_train)
            word_predictor = modules.WordPredictor(
                pred_embed, trainable=is_train)
            decoder_dim = L_DIM
        elif self.decoder_type == 'glove':
            pred_embed = self.glove_map
            word_predictor = modules.WordPredictor(
                pred_embed, trainable=is_train)
            decoder_dim = W_DIM
        elif self.decoder_type == 'dense':
            word_predictor = tf.layers.Dense(
                len(self.vocab['vocab']), use_bias=True, name='WordPredictor')
            decoder_dim = L_DIM

        logits_flat, pred_flat, pred_len_flat = modules.decode_L(
            in_L_flat, decoder_dim, self.glove_map, self.vocab['dict']['<s>'],
            unroll_type='teacher_forcing',
            seq=desc_flat, seq_len=desc_len_flat + 1,
            output_layer=word_predictor, is_train=is_train)

        _, greedy_flat, greedy_len_flat = modules.decode_L(
            in_L_flat, decoder_dim, self.glove_map, self.vocab['dict']['<s>'],
            unroll_type='greedy', end_token=self.vocab['dict']['<e>'],
            max_seq_len=self.data_cfg.max_len['region'] + 1,
            output_layer=word_predictor, is_train=is_train)

        with tf.name_scope('description_loss'):
            desc_used_mask = tf.sequence_mask(
                num_used_desc, maxlen=num_desc_box, dtype=tf.float32)

            loss = self.flat_description_loss(
                logits_flat, desc_flat, desc_len_flat + 1, desc_used_mask)
            pred_token_acc, pred_seq_acc = self.flat_description_accuracy(
                pred_flat, pred_flat_len, desc_flat, desc_flat_len + 1,
                desc_used_mask)
            greedy_token_acc, greedy_seq_acc = self.flat_description_accuracy(
                greedy_flat, greedy_flat_len, desc_flat, desc_flat_len + 1,
                desc_used_mask)

            self.losses['description'] = loss
            self.report['description_loss'] = loss
            self.report['pred_token_acc'] = pred_token_acc
            self.report['pred_seq_acc'] = pred_seq_acc
            self.report['greedy_token_acc'] = greedy_token_acc
            self.report['greedy_seq_acc'] = greedy_seq_acc

        # enc_L: encode object classes
        embed_seq = tf.nn.embedding_lookup(self.glove_all,
                                           self.batches['object']['objects'])
        enc_L_flat = modules.language_encoder(
            tf.reshape(embed_seq, [-1, self.object_max_name_len, W_DIM]),
            tf.reshape(self.batches['object']['objects_len'], [-1]),
            L_DIM, scope='language_encoder', reuse=tf.AUTO_REUSE)
        enc_L = tf.reshape(enc_L_flat, [-1, self.data_cfg.num_entry_box[key], L_DIM])
        if self.no_V_grad_enc_L: enc_L = tf.stop_gradient(enc_L)
        # L2V
        map_V = modules.L2V(enc_L, MAP_DIM, V_DIM, is_train=is_train,
                            scope='L2V', reuse=tf.AUTO_REUSE)
        # Classifier
        logits = modules.batch_word_classifier(
            feat_V, map_V, scope='object_classifier', reuse=tf.AUTO_REUSE)

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

        self.add_classification_summary(loss, acc, top_k_acc, image,
                                        pred_image, pred_string,
                                        name='object')
        return loss

    def add_string2image(self, image, string, prefix=None):
        def add_string2image_fn(batch_image, batch_string):
            import textwrap
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()
            new_images = []
            for image, string in zip(batch_image, batch_string):
                if prefix is not None:
                    string = prefix + string
                text_image = np.zeros([50, image.shape[1], 3], dtype=np.uint8)
                text_image += 220
                pil_text = Image.fromarray(text_image)
                pil_draw = ImageDraw.Draw(pil_text)
                for i, line in enumerate(textwrap.wrap(string, width=40)):
                    pil_draw.text((2, 2 + i * 15), line, font=font,
                                  fill=(10, 10, 50))
                text_image = np.array(pil_text).astype(np.float32) / 255.0
                image_with_text = np.concatenate([image, text_image], axis=0)
                new_images.append(image_with_text)
            return np.stack(new_images, axis=0)
        return tf.py_func(add_string2image_fn, [image, string], tf.float32)

    def build_region(self, is_train=True):
        # feat_V
        enc_I = modules.encode_I(self.batches['region']['image'],
                                 is_train=self.finetune_enc_I,
                                 reuse=tf.AUTO_REUSE)
        if not self.finetune_enc_I: enc_I = tf.stop_gradient(enc_I)
        feat_V = modules.I2V(enc_I, MAP_DIM, V_DIM, scope='I2V',
                             is_train=is_train, reuse=tf.AUTO_REUSE)
        # V2L [bs, L_DIM]
        map_L, V2L_hidden = modules.V2L(feat_V, MAP_DIM, L_DIM, scope='V2L',
                                        is_train=is_train, reuse=tf.AUTO_REUSE)
        # Language inputs
        seq = self.batches['region']['region_description']
        seq_len = self.batches['region']['region_description_len']
        wordset_seq = self.batches['region']['wordset_region_description']
        # enc_L: "enc" in language enc-dec [bs, L_DIM]
        embed_seq = tf.nn.embedding_lookup(self.glove_all, seq)
        enc_L = modules.language_encoder(
            embed_seq, seq_len, L_DIM,
            scope='language_encoder', reuse=tf.AUTO_REUSE)
        # dec_L: "dec" in language enc-dec + description [2*bs, L_DIM]
        dbl_seq = tf.concat([seq, seq], axis=0)
        dbl_seq_len = tf.concat([seq_len, seq_len], axis=0)
        dbl_start_tokens = tf.zeros([tf.shape(dbl_seq)[0]], dtype=tf.int32) + \
            self.vocab['dict']['<s>']
        dbl_seq_with_start = tf.concat([tf.expand_dims(dbl_start_tokens, axis=1),
                                        dbl_seq[:, :-1]], axis=1)
        dbl_embed_seq_with_start = tf.nn.embedding_lookup(self.glove_all,
                                                          dbl_seq_with_start)
        dbl_start_wordset_tokens = \
            tf.zeros([tf.shape(dbl_seq)[0]], dtype=tf.int32) + \
            self.wordset_vocab['dict']['<s>']
        in_L = tf.concat([feat_V, enc_L], axis=0)
        if self.use_embed_transform: decoder_dim = L_DIM
        else: decoder_dim = W_DIM
        logits, pred, pred_len = modules.language_decoder(
            in_L, dbl_embed_seq_with_start,
            dbl_seq_len + 1,  # seq_len + 1 is for <s>
            lambda e: tf.nn.embedding_lookup(self.glove_wordset, e),
            decoder_dim, dbl_start_wordset_tokens, self.wordset_vocab['dict']['<e>'],
            self.region_max_len + 1,  # + 1 for <e>
            unroll_type='teacher_forcing', output_layer=self.word_predictor,
            is_train=is_train, scope='language_decoder', reuse=tf.AUTO_REUSE)
        _, greedy_pred, greedy_pred_len = modules.language_decoder(
            in_L, dbl_embed_seq_with_start,
            dbl_seq_len + 1,  # seq_len + 1 is for <s>
            lambda e: tf.nn.embedding_lookup(self.glove_wordset, e),
            decoder_dim, dbl_start_wordset_tokens, self.wordset_vocab['dict']['<e>'],
            self.region_max_len + 1,  # + 1 for <e>
            unroll_type='greedy', output_layer=self.word_predictor,
            is_train=is_train, scope='language_decoder', reuse=tf.AUTO_REUSE)

        with tf.name_scope('LanguageGenerationLoss'):
            I2_logits, recon_logits = tf.split(logits, 2, axis=0)
            I2_pred, recon_pred = tf.split(pred, 2, axis=0)
            I2_pred_len, recon_pred_len = tf.split(pred_len, 2, axis=0)

            I2_greedy_pred, recon_greedy_pred = tf.split(greedy_pred, 2, axis=0)
            I2_greedy_pred_len, recon_greedy_pred_len = tf.split(
                greedy_pred_len, 2, axis=0)

            seq_mask = tf.sequence_mask(seq_len + 1, dtype=tf.float32)
            I2_loss = seq2seq.sequence_loss(
                I2_logits, wordset_seq, seq_mask, name='I2_sequence_loss')
            recon_loss = seq2seq.sequence_loss(
                recon_logits, wordset_seq, seq_mask, name='recon_sequence_loss')

            # Accuracy
            I2_token_acc, I2_seq_acc = self.sequence_accuracy(
                I2_pred, I2_pred_len, wordset_seq, seq_len + 1)
            recon_token_acc, recon_seq_acc = self.sequence_accuracy(
                recon_pred, recon_pred_len, wordset_seq, seq_len + 1)

            # Greedy accuracy
            I2_greedy_token_acc, I2_greedy_seq_acc = self.sequence_accuracy(
                I2_greedy_pred, I2_greedy_pred_len, wordset_seq, seq_len + 1)
            recon_greedy_token_acc, recon_greedy_seq_acc = self.sequence_accuracy(
                recon_greedy_pred, recon_greedy_pred_len, wordset_seq, seq_len + 1)

        with tf.name_scope('Summary'):
            gt_string = self.seq2string(wordset_seq, seq_len + 1)
            I2_string = self.seq2string(I2_pred, I2_pred_len)
            recon_string = self.seq2string(recon_pred, recon_pred_len)
            I2_greedy_string = self.seq2string(I2_greedy_pred,
                                               I2_greedy_pred_len)
            recon_greedy_string = self.seq2string(recon_greedy_pred,
                                                  recon_greedy_pred_len)

            image = self.batches['region']['image'] / 255.0
            image_n_gt_string = self.add_string2image(image, gt_string)
            image_n_I2_string = self.add_string2image(image, I2_string)
            image_n_recon_string = self.add_string2image(image, recon_string)
            image_n_I2_greedy_string = self.add_string2image(
                image, I2_greedy_string)
            image_n_recon_greedy_string = self.add_string2image(
                image, recon_greedy_string)
        # Debugging
        tf.summary.histogram('region_V2L_hidden1', V2L_hidden[0],
                             collections=['train'])
        tf.summary.histogram('region_V2L_hidden2', V2L_hidden[1],
                             collections=['train'])

        tf.summary.image('train-region_gt_string',
                         image_n_gt_string, max_outputs=10, collections=['train'])
        tf.summary.image('val-region_gt_string',
                         image_n_gt_string, max_outputs=10, collections=['val'])
        tf.summary.image('train-region_I2_string',
                         image_n_I2_string, max_outputs=10, collections=['train'])
        tf.summary.image('val-region_I2_string',
                         image_n_I2_string, max_outputs=10, collections=['val'])
        tf.summary.image('train-region_recon_string',
                         image_n_recon_string, max_outputs=10, collections=['train'])
        tf.summary.image('val-region_recon_string',
                         image_n_recon_string, max_outputs=10, collections=['val'])
        tf.summary.image('train-region_I2_greedy_string',
                         image_n_I2_greedy_string, max_outputs=10, collections=['train'])
        tf.summary.image('val-region_I2_greedy_string',
                         image_n_I2_greedy_string, max_outputs=10, collections=['val'])
        tf.summary.image('train-region_recon_greedy_string',
                         image_n_recon_greedy_string, max_outputs=10, collections=['train'])
        tf.summary.image('val-region_recon_greedy_string',
                         image_n_recon_greedy_string, max_outputs=10, collections=['val'])
        # Loss
        tf.summary.scalar('train-region/I2_loss',
                          I2_loss, collections=['train'])
        tf.summary.scalar('val-region/I2_loss',
                          I2_loss, collections=['val'])
        tf.summary.scalar('train-region/recon_loss',
                          recon_loss, collections=['train'])
        tf.summary.scalar('val-region/recon_loss',
                          recon_loss, collections=['val'])
        # Accuracy
        tf.summary.scalar('train-region/I2_token_acc',
                          I2_token_acc, collections=['train'])
        tf.summary.scalar('train-region/I2_seq_acc',
                          I2_seq_acc, collections=['train'])
        tf.summary.scalar('val-region/I2_token_acc',
                          I2_token_acc, collections=['val'])
        tf.summary.scalar('val-region/I2_seq_acc',
                          I2_seq_acc, collections=['val'])
        tf.summary.scalar('train-region/recon_token_acc',
                          recon_token_acc, collections=['train'])
        tf.summary.scalar('train-region/recon_seq_acc',
                          recon_seq_acc, collections=['train'])
        tf.summary.scalar('val-region/recon_token_acc',
                          recon_token_acc, collections=['val'])
        tf.summary.scalar('val-region/recon_seq_acc',
                          recon_seq_acc, collections=['val'])
        # Greedy accuracy
        tf.summary.scalar('train-region/I2_greedy_token_acc',
                          I2_greedy_token_acc, collections=['train'])
        tf.summary.scalar('train-region/I2_greedy_seq_acc',
                          I2_seq_acc, collections=['train'])
        tf.summary.scalar('val-region/I2_greedy_token_acc',
                          I2_token_acc, collections=['val'])
        tf.summary.scalar('val-region/I2_greedy_seq_acc',
                          I2_seq_acc, collections=['val'])
        tf.summary.scalar('train-region/recon_greedy_token_acc',
                          recon_token_acc, collections=['train'])
        tf.summary.scalar('train-region/recon_greedy_seq_acc',
                          recon_seq_acc, collections=['train'])
        tf.summary.scalar('val-region/recon_greedy_token_acc',
                          recon_token_acc, collections=['val'])
        tf.summary.scalar('val-region/recon_greedy_seq_acc',
                          recon_seq_acc, collections=['val'])
        # Text summary
        tf.summary.text('train-region_gt_string',
                        gt_string, collections=['train'])
        tf.summary.text('train-region_I2_string',
                        I2_string, collections=['train'])
        tf.summary.text('train-region_recon_string',
                        recon_string, collections=['train'])
        tf.summary.text('val-region_gt_string',
                        gt_string, collections=['val'])
        tf.summary.text('val-region_I2_string',
                        I2_string, collections=['val'])
        tf.summary.text('val-region_recon_string',
                        recon_string, collections=['val'])
        tf.summary.text('train-region_I2_greedy_string',
                        I2_greedy_string, collections=['train'])
        tf.summary.text('train-region_recon_greedy_string',
                        recon_greedy_string, collections=['train'])
        tf.summary.text('val-region_I2_greedy_string',
                        I2_greedy_string, collections=['val'])
        tf.summary.text('val-region_recon_greedy_string',
                        recon_greedy_string, collections=['val'])
        return I2_loss, recon_loss


    def build_blank_fill(self, is_train=True):
        # feat_V
        enc_I = modules.encode_I(self.batches['region']['image'],
                                 is_train=self.finetune_enc_I,
                                 reuse=tf.AUTO_REUSE)
        if not self.finetune_enc_I: enc_I = tf.stop_gradient(enc_I)
        feat_V = modules.I2V(enc_I, MAP_DIM, V_DIM, scope='I2V',
                             is_train=is_train, reuse=tf.AUTO_REUSE)
        # V2L [bs, L_DIM]
        map_L, V2L_hidden = modules.V2L(feat_V, MAP_DIM, L_DIM, scope='V2L',
                                        is_train=is_train, reuse=tf.AUTO_REUSE)
        # Language inputs
        seq_len = self.batches['region']['region_description_len']
        wordset_seq = self.batches['region']['wordset_region_description']
        blank_seq = self.batches['region']['blank_description']
        # enc_L: "enc" in language enc-dec [bs, L_DIM]
        embed_blank_seq = tf.nn.embedding_lookup(self.glove_wordset, blank_seq)
        enc_L = modules.language_encoder(
            embed_blank_seq, seq_len + 1, L_DIM,
            scope='language_encoder', reuse=tf.AUTO_REUSE)
        # dec_L: "dec" in language enc-dec + description [2*bs, L_DIM]
        start_tokens = tf.zeros([tf.shape(wordset_seq)[0]], dtype=tf.int32) + \
            self.wordset_vocab['dict']['<s>']
        seq_with_start = tf.concat([tf.expand_dims(start_tokens, axis=1),
                                    wordset_seq[:, :-1]], axis=1)
        embed_seq_with_start = tf.nn.embedding_lookup(self.glove_wordset,
                                                      seq_with_start)
        in_L = feat_V + enc_L
        if self.use_embed_transform: decoder_dim = L_DIM
        else: decoder_dim = W_DIM
        logits, pred, pred_len = modules.language_decoder(
            in_L, embed_seq_with_start,
            seq_len + 1,  # seq_len + 1 is for <s>
            lambda e: tf.nn.embedding_lookup(self.glove_wordset, e),
            decoder_dim, start_tokens, self.wordset_vocab['dict']['<e>'],
            self.region_max_len + 1,  # + 1 for <e>
            unroll_type='teacher_forcing', output_layer=self.word_predictor,
            is_train=is_train, scope='language_decoder', reuse=tf.AUTO_REUSE)
        _, greedy_pred, greedy_pred_len = modules.language_decoder(
            in_L, embed_seq_with_start,
            seq_len + 1,  # seq_len + 1 is for <s>
            lambda e: tf.nn.embedding_lookup(self.glove_wordset, e),
            decoder_dim, start_tokens, self.wordset_vocab['dict']['<e>'],
            self.region_max_len + 1,  # + 1 for <e>
            unroll_type='greedy', output_layer=self.word_predictor,
            is_train=is_train, scope='language_decoder', reuse=tf.AUTO_REUSE)

        with tf.name_scope('LanguageGenerationLoss'):
            seq_mask = tf.sequence_mask(seq_len + 1, dtype=tf.float32)
            loss = seq2seq.sequence_loss(
                logits, wordset_seq, seq_mask, name='blank_fill_sequence_loss')

            # Accuracy
            token_acc, seq_acc = self.sequence_accuracy(
                pred, pred_len, wordset_seq, seq_len + 1)

            # Greedy accuracy
            greedy_token_acc, greedy_seq_acc = self.sequence_accuracy(
                greedy_pred, greedy_pred_len, wordset_seq, seq_len + 1)

        with tf.name_scope('Summary'):
            gt_string = self.seq2string(wordset_seq, seq_len + 1)
            pred_string = self.seq2string(pred, pred_len)
            blank_string = self.seq2string(blank_seq, seq_len + 1)
            greedy_string = self.seq2string(greedy_pred, greedy_pred_len)

            image = self.batches['region']['image'] / 255.0
            image_gt = self.add_string2image(image, gt_string, prefix='gt: ')
            image_gt_blank = self.add_string2image(image_gt, blank_string,
                                                   prefix='blank: ')
            image_gt_blank_pred = self.add_string2image(
                image_gt_blank, pred_string, prefix='pred: ')
            image_gt_blank_pred_greedy = self.add_string2image(
                image_gt_blank_pred, greedy_string, prefix='greedy: ')

        # Debugging
        tf.summary.histogram('region_V2L_hidden1', V2L_hidden[0],
                             collections=['train'])
        tf.summary.histogram('region_V2L_hidden2', V2L_hidden[1],
                             collections=['train'])

        tf.summary.image('train-blank-fill_result',
                         image_gt_blank_pred_greedy, max_outputs=10,
                         collections=['train'])
        tf.summary.image('val-blank-fill_result',
                         image_gt_blank_pred_greedy, max_outputs=10,
                         collections=['val'])
        # Loss
        tf.summary.scalar('train-blank-fill/loss',
                          loss, collections=['train'])
        tf.summary.scalar('val-blank-fill/loss',
                          loss, collections=['val'])
        # Accuracy
        tf.summary.scalar('train-blank-fill/token_acc',
                          token_acc, collections=['train'])
        tf.summary.scalar('train-blank-fill/seq_acc',
                          seq_acc, collections=['train'])
        tf.summary.scalar('val-blank-fill/token_acc',
                          token_acc, collections=['val'])
        tf.summary.scalar('val-blank-fill/seq_acc',
                          seq_acc, collections=['val'])

        # Greedy accuracy
        tf.summary.scalar('train-blank-fill/greedy_token_acc',
                          greedy_token_acc, collections=['train'])
        tf.summary.scalar('train-blank-fill/greedy_seq_acc',
                          greedy_seq_acc, collections=['train'])
        tf.summary.scalar('val-blank-fill/greedy_token_acc',
                          greedy_token_acc, collections=['val'])
        tf.summary.scalar('val-blank-fill/greedy_seq_acc',
                          greedy_seq_acc, collections=['val'])
        return loss


    def build(self, is_train=True):
        self.v_loss = 0
        self.l_loss = 0

        if not self.no_object:
            log.info('build object')
            object_v_loss = self.build_object(is_train=is_train)
            self.v_loss += object_v_loss

        if not self.no_region:
            log.info('build region')
            region_v_loss, region_l_loss = self.build_region(is_train=is_train)
            self.v_loss += region_v_loss
            self.l_loss += region_l_loss

        if self.use_blank_fill:
            log.info('build blank fill')
            self.v_loss += self.build_blank_fill(is_train=is_train)

        self.loss = self.v_loss + self.l_loss
        tf.summary.scalar('train/loss', self.loss, collections=['train'])
        tf.summary.scalar('val/loss', self.loss, collections=['val'])
        tf.summary.scalar('train/v_loss', self.v_loss, collections=['train'])
        tf.summary.scalar('val/v_loss', self.v_loss, collections=['val'])
        tf.summary.scalar('train/l_loss', self.l_loss, collections=['train'])
        tf.summary.scalar('val/l_loss', self.l_loss, collections=['val'])

        self.report['loss'] = self.loss
