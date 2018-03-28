import json
import h5py
import os
import numpy as np
import tensorflow as tf

from util import box_utils, log
from vlmap import modules

W_DIM = 300  # Word dimension
L_DIM = 512  # Language dimension
MAP_DIM = 512
V_DIM = 512
ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'


class Model(object):

    def __init__(self, batch, config, is_train=True):
        self.batch = batch
        self.config = config
        self.image_dir = config.image_dir

        self.losses = {}
        self.report = {}
        self.mid_result = {}
        self.vis_image = {}

        self.vocab = json.load(open(config.vocab_path, 'r'))
        self.glove_map = modules.GloVe_vocab(self.vocab)

        # answer candidates
        log.infov('loading answer candidates...')
        with h5py.File(os.path.join(config.tf_record_dir,
                                    'data_info.hdf5'), 'r') as f:
            self.answer_intseq_value = f['data_info']['intseq_ans'].value
            self.answer_intseq_len_value = f['data_info']['intseq_ans_len'].value
            self.num_answer = int(f['data_info']['num_answers'].value)

            with tf.device('/cpu:0'):
                self.answer_intseq = tf.convert_to_tensor(
                    self.answer_intseq_value, dtype=tf.int32,
                    name='answer_intseq')  # [num_answer, len]
                self.answer_intseq_len = tf.convert_to_tensor(
                    self.answer_intseq_len_value, dtype=tf.int32,
                    name='answer_intseq_len')  # [num_answer]
        log.infov('done')

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
        log.infov('done')

        self.build(is_train=is_train)

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

    def visualize_vqa_result(self,
                             image_id, box, num_box,
                             att_score,
                             q_intseq, q_intseq_len,
                             label, pred, line_width=2):
        def construct_visualization(b_image_id, bb_box, b_num_box,
                                    bb_att_score,
                                    b_q_intseq, b_q_intseq_len,
                                    b_label, b_pred):
            # b_ : batch
            # bb_ : [batch, description]
            import textwrap
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()

            def intseq2str(intseq):
                return ' '.join([self.vocab['vocab'][i] for i in intseq])

            def string2image(string, image_width=1080):
                pil_text = Image.fromarray(
                    np.zeros([15, image_width, 3], dtype=np.uint8) + 220)
                t_draw = ImageDraw.Draw(pil_text)
                for l, line in enumerate(textwrap.wrap(string, width=90)):
                    t_draw.text((2, 2 + l * 15), line, font=font,
                                fill=(10, 10, 50))
                return np.array(pil_text).astype(np.uint8)

            batch_vis_image = []
            for batch_idx, image_id in enumerate(b_image_id):
                image_path = os.path.join(self.image_dir,
                                          image_id.replace('-', '/'))
                image = Image.open(image_path)
                image = np.array(image.resize([540, 540]).convert('RGB'))
                float_image = image.astype(np.float32)
                att_mask = np.zeros_like(float_image)

                b_score = bb_att_score[batch_idx]
                max_score_idx = np.argmax(b_score)
                for box_idx in range(b_num_box[batch_idx]):
                    box = bb_box[batch_idx][box_idx]
                    att_mask = box_utils.add_value_x1y1x2y2(
                        image=att_mask, box=box, value=b_score[box_idx])
                att_image = Image.fromarray(
                        (float_image * att_mask).astype(np.uint8))
                draw = ImageDraw.Draw(att_image)
                (x1, y1, x2, y2) = bb_box[batch_idx][max_score_idx]
                for w in range(line_width):
                    draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                   outline=(255, 0, 0))
                att_image = np.array(att_image, dtype=np.uint8)

                vis_image = np.concatenate([image, att_image], axis=1)
                width = vis_image.shape[1]

                q_intseq = b_q_intseq[batch_idx]
                q_intseq_len = b_q_intseq_len[batch_idx]

                question_str = ' '.join(
                    [self.vocab['vocab'][t] for t in q_intseq[:q_intseq_len]])
                question_image = string2image('Q: ' + question_str,
                                              image_width=width)
                label = b_label[batch_idx]
                if label < self.num_answer:
                    a_intseq = self.answer_intseq_value[label]
                    a_intseq_len = self.answer_intseq_len_value[label]
                    label_answer_str = ' '.join(
                        [self.vocab['vocab'][t] for t in a_intseq[:a_intseq_len]])
                else:
                    label_answer_str = '<infrequent answer>'
                label_answer_image = string2image('GT: ' + label_answer_str,
                                                  image_width=width)

                pred = b_pred[batch_idx]
                a_intseq = self.answer_intseq_value[pred]
                a_intseq_len = self.answer_intseq_len_value[pred]
                pred_answer_str = ' '.join(
                    [self.vocab['vocab'][t] for t in a_intseq[:a_intseq_len]])
                pred_answer_image = string2image('PRED: ' + pred_answer_str,
                                                 image_width=width)

                vis_image = np.concatenate(
                    [vis_image, question_image,
                     label_answer_image, pred_answer_image], axis=0)
                batch_vis_image.append(vis_image)
            batch_vis_image = np.stack(batch_vis_image, axis=0)
            return batch_vis_image
        return tf.py_func(
            construct_visualization,
            inp=[image_id, box, num_box, att_score,
                 q_intseq, q_intseq_len, label, pred],
            Tout=tf.uint8)

    def build(self, is_train=True):
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
            V_ft.set_shape([None, 36, 2048])
            num_V_ft = tf.gather(self.num_boxes, self.batch['image_idx'],
                                 name='gather_num_V_ft', axis=0)

        log.warning('v_linear_v')
        v_linear_v = modules.fc_layer(
            V_ft, V_DIM, use_bias=True, use_bn=False,
            activation_fn=tf.tanh, is_training=is_train,
            scope='v_linear_v')

        """
        Encode question
        """
        q_embed = tf.nn.embedding_lookup(self.glove_map, self.batch['q_intseq'])
        # [bs, L_DIM]
        q_L_ft = modules.encode_L(q_embed, self.batch['q_intseq_len'], L_DIM)

        # [bs, V_DIM}
        log.warning('q_linear_v')
        q_linear_v = modules.fc_layer(
            q_L_ft, V_DIM, use_bias=True, use_bn=False,
            activation_fn=tf.tanh, is_training=is_train,
            scope='q_linear_v')

        """
        Perform attention
        """
        att_score = modules.hadamard_attention(v_linear_v, num_V_ft, q_linear_v,
                                               is_train=is_train)
        self.mid_result['att_score'] = att_score
        pooled_V_ft = modules.attention_pooling(V_ft, att_score)

        """
        Answer classification
        """
        # perform two layer feature encoding and predict output
        with tf.variable_scope('reasoning') as scope:
            log.warning(scope.name)
            # [bs, L_DIM]
            log.warning('pooled_linear_l')
            pooled_linear_l = modules.fc_layer(
                pooled_V_ft, L_DIM, use_bias=True, use_bn=False,
                activation_fn=tf.tanh, is_training=is_train,
                scope='pooled_linear_l')

            log.warning('q_linear_l')
            q_linear_l = modules.fc_layer(
                q_L_ft, L_DIM, use_bias=True, use_bn=False,
                activation_fn=tf.tanh, is_training=is_train,
                scope='q_linear_l')

            logit = modules.fc_layer(
                pooled_linear_l * q_linear_l, self.num_answer,
                use_bias=True, use_bn=False,
                activation_fn=None, is_training=is_train, scope='classifier')

        """
        Compute loss and accuracy
        """
        with tf.name_scope('loss'):
            answer_target = self.batch['answer_target']
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=answer_target, logits=logit)
            loss = tf.reduce_mean(loss)
            pred = tf.cast(tf.argmax(logit, axis=-1), dtype=tf.int32)
            one_hot_pred = tf.one_hot(pred, depth=self.num_answer,
                                      dtype=tf.float32)
            acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target, axis=-1))

            self.mid_result['pred'] = pred

            self.losses['answer'] = loss
            self.report['answer_loss'] = loss
            self.report['answer_accuracy'] = acc

        """
        Prepare image summary
        """
        """
        with tf.name_scope('prepare_summary'):
            self.vis_image['image_attention_qa'] = self.visualize_vqa_result(
                self.batch['image_id'], self.batch['box'], self.batch['num_box'],
                self.mid_result['att_score'],
                self.batch['q_intseq'], self.batch['q_intseq_len'],
                self.batch['answer_id'], self.mid_result['pred'],
                line_width=2)
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
