import json
import numpy as np
import tensorflow as tf

from util import log
from vlmap import modules

TOP_K = 5
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
        self.glove_map = modules.GloVe(config.glove_path)

        # model parameters
        self.ft_enc_I = config.ft_enc_I
        self.no_V_grad_enc_L = config.no_V_grad_enc_L
        self.use_relation = config.use_relation
        self.description_task = config.description_task
        self.decoder_type = config.decoder_type
        self.num_aug_retrieval = max(config.num_aug_retrieval, 0)

        self.target_entry = ['object', 'attribute']
        if self.use_relation: self.target_entry.append('relationship')

        if self.decoder_type == 'glove_et':
            pred_embed = modules.embed_transform(
                self.glove_map, L_DIM, L_DIM, is_train=is_train)
            self.word_predictor = modules.WordPredictor(
                pred_embed, trainable=is_train)
            self.decoder_dim = L_DIM
        elif self.decoder_type == 'glove':
            pred_embed = self.glove_map
            self.word_predictor = modules.WordPredictor(
                pred_embed, trainable=is_train)
            self.decoder_dim = W_DIM
        elif self.decoder_type == 'dense':
            self.word_predictor = tf.layers.Dense(
                len(self.vocab['vocab']), use_bias=True, name='WordPredictor')
            self.decoder_dim = L_DIM

        self.build(is_train=is_train)

    def filter_vars(self, all_vars):
        enc_I_vars = []
        learn_v_vars = []  # variables learning from vision loss
        learn_l_vars = []  # variables learning from language loss
        for var in all_vars:
            if var.name.split('/')[0] == 'resnet_v1_50':
                enc_I_vars.append(var)
                if self.ft_enc_I:
                    learn_v_vars.append(var)
                    learn_l_vars.append(var)
            elif var.name.split('/')[0] == 'encode_L':
                learn_l_vars.append(var)
                if not self.no_V_grad_enc_L:
                    learn_v_vars.append(var)
            elif var.name.split('/')[0] == 'decode_L':
                learn_l_vars.append(var)
            else:
                learn_v_vars.append(var)
                learn_l_vars.append(var)
        return enc_I_vars, learn_v_vars, learn_l_vars

    def get_enc_I_param_path(self):
        return ENC_I_PARAM_PATH

    def vis_description(self, image, box, desc, desc_len,
                        blank_desc, blank_desc_len, is_blank_used,
                        pred, pred_len, greedy, greedy_len, used_mask,
                        vis_numbox, line_width):
        def add_result2image_fn(b_image, bb_box, bb_desc, bb_desc_len,
                                bb_blank_desc, bb_blank_desc_len,
                                bb_pred, bb_pred_len, bb_greedy, bb_greedy_len,
                                bb_used_mask):
            # b_ : batch
            # bb_ : [batch, description]
            import textwrap
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()

            def intseq2str(intseq):
                return ' '.join([self.vocab['vocab'][i] for i in intseq])

            def string2image(string):
                pil_text = Image.fromarray(
                    np.zeros([35, b_image.shape[1], 3], dtype=np.uint8) + 220)
                t_draw = ImageDraw.Draw(pil_text)
                for l, line in enumerate(textwrap.wrap(string, width=90)):
                    t_draw.text((2, 2 + l * 15), line, font=font,
                                fill=(10, 10, 50))
                return np.array(pil_text).astype(np.float32) / 255.0
            bb_image_with_text = []
            for b, image in enumerate(b_image):
                b_image_with_text = []
                for i in range(min(vis_numbox, len(bb_box[b]))):
                    if bb_used_mask[b][i] == 0:
                        vis_images = []
                        vis_images.append(image.copy() / 255.0)
                        vis_images.append(string2image('[{}] [not_used]'.format(i)))
                        vis_images.append(string2image(' '))  # blank
                        vis_images.append(string2image(' '))  # pred
                        vis_images.append(string2image(' '))  # greedy
                        b_image_with_text.append(
                            np.concatenate(vis_images, axis=0))
                        continue

                    pil_image = Image.fromarray(image.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_image)
                    (x1, y1, x2, y2) = bb_box[b][i]
                    for w in range(line_width):
                        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                       outline=(255, 0, 0))
                    vis_images = []
                    vis_images.append(
                        np.array(pil_image).astype(np.float32) / 255.0)
                    gt_string = intseq2str(
                        bb_desc[b][i][:bb_desc_len[b][i]])
                    blank_string = intseq2str(
                        bb_blank_desc[b][i][:bb_blank_desc_len[b][i]])
                    blank_string = ('[used] ' if is_blank_used
                                    else '[not_used] ') + blank_string
                    pred_string = intseq2str(
                        bb_pred[b][i][:bb_pred_len[b][i]])
                    greedy_string = intseq2str(
                        bb_greedy[b][i][:bb_greedy_len[b][i]])
                    vis_images.append(string2image('[GT]: ' + gt_string))
                    vis_images.append(string2image('[Blank]: ' + blank_string))
                    vis_images.append(string2image('[Pred]: ' + pred_string))
                    vis_images.append(string2image('[Greedy]: ' + greedy_string))
                    b_image_with_text.append(np.concatenate(vis_images, axis=0))
                num_i = len(b_image_with_text)
                if num_i < vis_numbox:
                    b_image_with_text.extend(
                        [b_image_with_text[-1]] * (vis_numbox - num_i))
                bb_image_with_text.append(
                    np.concatenate(b_image_with_text, axis=1))
            return np.stack(bb_image_with_text, axis=0)
        return tf.py_func(
            add_result2image_fn,
            inp=[image, box, desc, desc_len, blank_desc, blank_desc_len,
                 pred, pred_len, greedy, greedy_len, used_mask],
            Tout=tf.float32)

    def vis_retrieval_I(self, image, ir_visbox, desc, desc_len,
                        num_aug, ir_num_k, ir_logits, ir_gt, used_mask,
                        vis_numbox, line_width):
        ir_gt_idx = tf.cast(tf.argmax(ir_gt, axis=-1), tf.int32)
        probs = tf.nn.softmax(ir_logits, axis=-1)
        top_k_prob, top_k_pred = tf.nn.top_k(probs, k=TOP_K)

        def add_result2image_fn(b_image, bb_ir_box, bb_desc, bb_desc_len,
                                bb_gt_idx, bb_top_k_prob, bb_top_k_pred,
                                bb_used_mask):
            # b_ : batch
            # bb_ : [batch, description]
            import textwrap
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()
            def intseq2str(intseq):
                return ' '.join([self.vocab['vocab'][i] for i in intseq])

            def string2image(string):
                pil_text = Image.fromarray(
                    np.zeros([35 * (num_aug + 1), b_image.shape[1] * (num_aug + 1), 3],
                             dtype=np.uint8) + 220)
                t_draw = ImageDraw.Draw(pil_text)
                for l, line in enumerate(textwrap.wrap(string, width=90)):
                    t_draw.text((2, 2 + l * 15), line, font=font, fill=(10, 10, 50))
                return np.array(pil_text).astype(np.float32) / 255.0
            b_image_with_box = []
            for image, b_ir_box, b_desc in zip(b_image, bb_ir_box, bb_desc):
                desc_image_with_box = []
                for i in range(min(vis_numbox, len(b_desc))):
                    pil_image = Image.fromarray(image.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_image)
                    for ir_box in b_ir_box[i]:
                        (x1, y1, x2, y2) = ir_box
                        for w in range(line_width):
                            draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                           outline=(0, 0, 0))
                    desc_image_with_box.append(np.array(pil_image))
                b_image_with_box.append(desc_image_with_box)

            bb_image_with_text = []
            for b, (image, b_ir_box, b_desc, b_desc_len, b_gt_idx, b_used_mask) \
                in enumerate(zip(b_image, bb_ir_box, bb_desc, bb_desc_len,
                                 bb_gt_idx, bb_used_mask)):
                b_image_with_text = []
                for i in range(min(vis_numbox, len(b_desc))):
                    if b_used_mask[i] == 0:
                        box_vis_img = np.zeros(
                            [image.shape[0], image.shape[1] * (1 + num_aug), 3],
                            dtype=np.float32)
                        string_img = string2image('[usused]')
                        b_image_with_text.append(
                            np.concatenate([box_vis_img, string_img], axis=0))
                        continue

                    target_images = []
                    target_images.append(b_image_with_box[b][i])
                    for j in range(num_aug):
                        target_images.append(b_image_with_box[b - j - 1][i])
                    gt_idx = b_gt_idx[i]
                    gt_b = int(gt_idx / ir_num_k)
                    gt_i = int(gt_idx) % ir_num_k
                    gt_box = bb_ir_box[b - gt_b, i, gt_i]
                    gt_img = target_images[gt_b]
                    pil_gt_img = Image.fromarray(gt_img.copy())
                    gt_draw = ImageDraw.Draw(pil_gt_img)
                    (x1, y1, x2, y2) = gt_box
                    for w in range(line_width):
                        gt_draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                          outline=(0, 0, 255))
                    gt_draw.text((x1 + 2, y1 + 2), 'GT', font=font,
                                 fill=(0, 0, 255))
                    target_images[gt_b] = np.array(pil_gt_img)
                    for k in range(TOP_K):
                        prob = bb_top_k_prob[b, i, k]
                        pred = bb_top_k_pred[b, i, k]
                        if pred == gt_idx: color = (0, 255 - (k * 20), 0)
                        else: color = (255 - (k * 20), 0, 0)
                        p_b = int(pred / ir_num_k)
                        p_i = int(pred) % ir_num_k
                        p_box = bb_ir_box[b - p_b, i, p_i]
                        p_img = target_images[p_b]
                        pil_p_img = Image.fromarray(p_img.copy())
                        p_draw = ImageDraw.Draw(pil_p_img)
                        (x1, y1, x2, y2) = p_box
                        for w in range(line_width):
                            p_draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                             outline=color)
                        p_draw.text((x1 + 2, y1 + 2),
                                    'Top-{} ({:.5f})'.format(k, prob),
                                    font=font, fill=color)
                        target_images[p_b] = np.array(pil_p_img)
                    box_vis_img = np.concatenate(
                        target_images, axis=1).astype(np.float32) / 255.0
                    description = intseq2str(b_desc[i][:b_desc_len[i]])
                    string_img = string2image('desc: {}'.format(description))
                    b_image_with_text.append(
                        np.concatenate([box_vis_img, string_img], axis=0))
                num_i = len(b_image_with_text)
                if num_i < vis_numbox:
                    b_image_with_text.extend(
                        [b_image_with_text[-1]] * (vis_numbox - num_i))
                bb_image_with_text.append(
                    np.concatenate(b_image_with_text, axis=0))
            output = np.stack(bb_image_with_text, axis=0)
            return np.stack(bb_image_with_text, axis=0)
        return tf.py_func(
            add_result2image_fn,
            inp=[image, ir_visbox, desc, desc_len, ir_gt_idx,
                 top_k_prob, top_k_pred, used_mask],
            Tout=tf.float32)

    def vis_retrieval_L(self, image, visbox, desc, desc_len,
                        lr_desc_idx, lr_num_k, lr_logits, lr_gt, used_mask,
                        vis_numbox, line_width):
        lr_gt_idx = tf.cast(tf.argmax(lr_gt, axis=-1), tf.int32)
        probs = tf.nn.softmax(lr_logits, axis=-1)
        top_k_prob, top_k_pred = tf.nn.top_k(probs, k=TOP_K)

        def add_result2image_fn(b_image, bb_box, bb_desc, bb_desc_len,
                                bb_desc_idx, bb_gt_idx,
                                bb_top_k_prob, bb_top_k_pred, bb_used_mask):
            # b_ : batch
            # bb_ : [batch, box]
            import textwrap
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()

            def intseq2str(intseq):
                return ' '.join([self.vocab['vocab'][i] for i in intseq])

            def string2image(string):
                pil_text = Image.fromarray(
                    np.zeros([35, b_image.shape[1], 3], dtype=np.uint8) + 220)
                t_draw = ImageDraw.Draw(pil_text)
                for l, line in enumerate(textwrap.wrap(string, width=90)):
                    t_draw.text((2, 2 + l * 15), line, font=font,
                                fill=(10, 10, 50))
                return np.array(pil_text).astype(np.float32) / 255.0

            bb_image_with_text = []
            for b, (image, b_box, b_gt_idx, b_top_k_prob,
                    b_top_k_pred, b_used_mask) \
                in enumerate(zip(b_image, bb_box, bb_gt_idx, bb_top_k_prob,
                                 bb_top_k_pred, bb_used_mask)):
                b_image_with_text = []
                for i in range(min(vis_numbox, len(b_box))):
                    if b_used_mask[i] == 0:
                        vis_images = []
                        vis_images.append(image.copy() / 255.0)
                        vis_images.append(string2image('[{}] [unused]'.format(i)))
                        for _ in range(TOP_K):
                            vis_images.append(string2image('   '))
                        b_image_with_text.append(
                            np.concatenate(vis_images, axis=0))
                        continue

                    pil_image = Image.fromarray(image.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_image)
                    (x1, y1, x2, y2) = b_box[i]
                    for w in range(line_width):
                        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                       outline=(255, 0, 0))
                    vis_images = []
                    vis_images.append(
                        np.array(pil_image).astype(np.float32) / 255.0)
                    gt_i = bb_desc_idx[b, i, b_gt_idx[i]]
                    gt_intseq = bb_desc[b, gt_i][:bb_desc_len[b, gt_i]]
                    gt_string = 'gt: {}'.format(intseq2str(gt_intseq))
                    vis_images.append(string2image(gt_string))
                    for k, pred in enumerate(b_top_k_pred[i]):
                        p_b = b - int(pred / lr_num_k)
                        p_i = int(pred % lr_num_k)
                        pred_i = bb_desc_idx[p_b, i, p_i]
                        pred_intseq = bb_desc[p_b, pred_i][:bb_desc_len[p_b, pred_i]]
                        pred_string = intseq2str(pred_intseq)
                        pred_string = 'pred ({:.5f}): {}'.format(
                            b_top_k_prob[i][k], pred_string)
                        vis_images.append(string2image(pred_string))
                    b_image_with_text.append(np.concatenate(vis_images, axis=0))
                num_i = len(b_image_with_text)
                if num_i < vis_numbox:
                    b_image_with_text.extend(
                        [b_image_with_text[-1]] * (vis_numbox - num_i))
                bb_image_with_text.append(
                    np.concatenate(b_image_with_text, axis=1))
            return np.stack(bb_image_with_text, axis=0)
        return tf.py_func(
            add_result2image_fn,
            inp=[image, visbox, desc, desc_len, lr_desc_idx, lr_gt_idx,
                 top_k_prob, top_k_pred, used_mask],
            Tout=tf.float32)

    def vis_binary_image_classification(self, image, visbox, logits, labels,
                                        intseqs, intseqs_len, names,
                                        used_mask, vis_numbox, line_width):
        num_label = tf.cast(tf.reduce_sum(labels, axis=-1), tf.int32)
        probs = tf.nn.sigmoid(logits)
        top_k_prob, top_k_pred = tf.nn.top_k(probs, k=TOP_K)

        def add_result2image_fn(b_image, bb_box, bb_num_label,
                                bb_top_k_prob, bb_top_k_pred,
                                bb_intseqs, bb_intseqs_len, bb_names,
                                bb_used_mask):
            # b_ : batch
            # bb_ : [batch, box]
            import textwrap
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()
            bb_image_with_text = []
            for image, b_box, b_num_label, b_top_k_prob, b_top_k_pred,\
                b_intseqs, b_intseqs_len, b_names,\
                b_used_mask in zip(b_image, bb_box, bb_num_label,
                                   bb_top_k_prob, bb_top_k_pred,
                                   bb_intseqs, bb_intseqs_len,
                                   bb_names, bb_used_mask):
                b_image_with_text = []
                for i in range(min(vis_numbox, len(b_box))):
                    # Draw box and attach text image
                    pil_image = Image.fromarray(image.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_image)
                    (x1, y1, x2, y2) = b_box[i]
                    for w in range(line_width):
                        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                       outline=(255, 0, 0))
                    box_image = np.array(pil_image).astype(np.float32) / 255.0

                    text_image = np.zeros([35, image.shape[1], 3],
                                          dtype=np.uint8) + 220
                    pil_text = Image.fromarray(text_image)
                    t_draw = ImageDraw.Draw(pil_text)
                    intseqs = b_intseqs[i]
                    intseqs_len = b_intseqs_len[i]
                    names = b_names[i]
                    string = ''
                    if not b_used_mask[i]: string += '[not_used], '
                    string += '[gt]: ({}) '.format(b_num_label[i])
                    for n in range(b_num_label[i]):
                        string += '{}, '.format(
                            ' '.join([self.vocab['vocab'][t]
                                      for t in
                                      intseqs[n][:intseqs_len[n]]]))
                    string += '[pred]: '
                    for prob, pred in zip(b_top_k_prob[i], b_top_k_pred[i]):
                        string += '{}'.format(
                            ' '.join([self.vocab['vocab'][t]
                                      for t in
                                      intseqs[pred][:intseqs_len[pred]]]))
                        string += '({:.5f}), '.format(prob)
                    for l, line in enumerate(textwrap.wrap(string, width=90)):
                        t_draw.text((2, 2 + l * 15), line, font=font,
                                    fill=(10, 10, 50))
                    text_image = np.array(pil_text).astype(np.float32) / 255.0

                    image_with_text = np.concatenate([box_image, text_image],
                                                     axis=0)
                    b_image_with_text.append(image_with_text)
                num_i = len(b_image_with_text)
                if num_i < vis_numbox:
                    b_image_with_text.extend(
                        [b_image_with_text[-1]] * (vis_numbox - num_i))
                bb_image_with_text.append(
                    np.concatenate(b_image_with_text, axis=1))
            return np.stack(bb_image_with_text, axis=0)
        return tf.py_func(
            add_result2image_fn,
            inp=[image, visbox, num_label, top_k_prob,
                 top_k_pred, intseqs, intseqs_len, names, used_mask],
            Tout=tf.float32)




    def vis_n_way_image_classification(self, image, visbox, logits, labels,
                                       intseqs, intseqs_len, names,
                                       used_mask, vis_numbox, line_width):
        label_token = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        probs = tf.nn.softmax(logits, axis=-1)
        top_k_prob, top_k_pred = tf.nn.top_k(probs, k=TOP_K)

        def add_result2image_fn(b_image, bb_box, bb_label,
                                bb_top_k_prob, bb_top_k_pred,
                                bb_intseqs, bb_intseqs_len, bb_names,
                                bb_used_mask):
            # b_ : batch
            # bb_ : [batch, box]
            import textwrap
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()
            bb_image_with_text = []
            for image, b_box, b_label, b_top_k_prob, b_top_k_pred,\
                b_intseqs, b_intseqs_len, b_names,\
                b_used_mask in zip(b_image, bb_box, bb_label,
                                   bb_top_k_prob, bb_top_k_pred,
                                   bb_intseqs, bb_intseqs_len,
                                   bb_names, bb_used_mask):
                b_image_with_text = []
                for i in range(min(vis_numbox, len(b_box))):
                    # Draw box and attach text image
                    pil_image = Image.fromarray(image.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_image)
                    (x1, y1, x2, y2) = b_box[i]
                    for w in range(line_width):
                        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                       outline=(255, 0, 0))
                    box_image = np.array(pil_image).astype(np.float32) / 255.0

                    text_image = np.zeros([35, image.shape[1], 3],
                                          dtype=np.uint8) + 220
                    pil_text = Image.fromarray(text_image)
                    t_draw = ImageDraw.Draw(pil_text)
                    intseqs = b_intseqs[i]
                    intseqs_len = b_intseqs_len[i]
                    names = b_names[i]
                    string = '{}, [gt]: {}'.format(
                        '[used]' if b_used_mask[i] else '[not_used]',
                        ' '.join([self.vocab['vocab'][t]
                                  for t in
                                  intseqs[b_label[i]][:intseqs_len[b_label[i]]]]))
                    string += ', [pred]: '
                    for prob, pred in zip(b_top_k_prob[i], b_top_k_pred[i]):
                        string += '{}'.format(
                            ' '.join([self.vocab['vocab'][t]
                                      for t in
                                      intseqs[pred][:intseqs_len[pred]]]))
                        string += '({:.5f}), '.format(prob)
                    for l, line in enumerate(textwrap.wrap(string, width=90)):
                        t_draw.text((2, 2 + l * 15), line, font=font,
                                    fill=(10, 10, 50))
                    text_image = np.array(pil_text).astype(np.float32) / 255.0

                    image_with_text = np.concatenate([box_image, text_image],
                                                     axis=0)
                    b_image_with_text.append(image_with_text)
                num_i = len(b_image_with_text)
                if num_i < vis_numbox:
                    b_image_with_text.extend(
                        [b_image_with_text[-1]] * (vis_numbox - num_i))
                bb_image_with_text.append(
                    np.concatenate(b_image_with_text, axis=1))
            return np.stack(bb_image_with_text, axis=0)
        return tf.py_func(
            add_result2image_fn,
            inp=[image, visbox, label_token, top_k_prob,
                 top_k_pred, intseqs, intseqs_len, names, used_mask],
            Tout=tf.float32)

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

    def binary_classification_loss(self, logits, labels, mask=None):
        # Loss
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        cross_entropy = tf.reduce_mean(cross_entropy, axis=-1)
        if mask is None:
            loss = tf.reduce_mean(cross_entropy)
        else:
            loss = tf.reduce_sum(cross_entropy * mask) / tf.reduce_sum(mask)

        binary_pred = tf.cast(logits > 0, tf.int32)
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

        return loss, acc, recall, precision

    def flat_description_loss(self, logits_flat,
                              desc_flat, desc_len_flat, used_mask):
        desc_maxlen = tf.shape(desc_flat)[-1]
        used_mask_flat = tf.expand_dims(tf.reshape(used_mask, [-1]), axis=-1)
        mask = tf.sequence_mask(
            desc_len_flat, maxlen=desc_maxlen, dtype=tf.float32) * used_mask_flat
        # dynamic_padding logit
        sz = tf.shape(logits_flat)
        pad = tf.zeros([sz[0], desc_maxlen - sz[1], sz[2]],
                       dtype=logits_flat.dtype)
        logits_flat = tf.concat([logits_flat, pad], axis=1)
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
        mask_int = tf.cast(tf.sequence_mask(
            desc_len_flat, maxlen=max_len,
            dtype=tf.float32, name='mask') * used_mask_flat, dtype=tf.int32)
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
        used_mask_flat = tf.squeeze(used_mask_flat, axis=-1)
        seq_acc = tf.reduce_sum(tf.to_float(seq_acc) * used_mask_flat) / \
            tf.reduce_sum(used_mask_flat)
        return token_acc, seq_acc

    def aug_retrieval(self, V, gt, num_aug, num_b, num_k, dim,
                      scope='aug_retrieval'):
        with tf.name_scope(scope):
            sz = tf.shape(V)
            rotate_idx = [tf.mod(tf.range(-1 - i, sz[0] - 1 - i), sz[0])
                          for i in range(num_aug)]
            rotate_idx = tf.stack(rotate_idx, axis=0)
            rotate_V = tf.transpose(tf.reshape(
                tf.gather(V, tf.reshape(rotate_idx, [-1]), axis=0),
                [num_aug, -1, num_b, num_k, dim]), [1, 2, 0, 3, 4])
            rotate_V = tf.reshape(rotate_V, [-1, num_b, num_k * num_aug, dim])
            V = tf.concat([V, rotate_V], axis=2)
            gt_pad = tf.zeros([sz[0], num_b, num_k * num_aug], dtype=tf.float32)
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
        visbox_flat = tf.reshape(self.batch['box'], [-1, 4])  # box for visualization

        # _flat regards "num box" dimension as a batch
        roi_ft_flat = tf.reshape(roi_ft, [-1, ROI_SZ, ROI_SZ, V_DIM])

        V_ft_flat = modules.I2V(roi_ft_flat, V_DIM, V_DIM, is_train=is_train)

        """
        Classification: object, attribute, relationship
        """
        for key in self.target_entry:
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
            box_idx = modules.batch_box(self.batch['{}_box_idx'.format(key)],
                                        offset=self.data_cfg.num_box)
            box_V_ft_flat = tf.gather(V_ft_flat, tf.reshape(box_idx, [-1]))
            box_V_ft = tf.reshape(box_V_ft_flat, [-1, num_b, V_DIM])
            self.mid_result['{}_visbox'.format(key)] = tf.reshape(tf.gather(
                visbox_flat, tf.reshape(box_idx, [-1])), [-1, num_b, 4])

            # classification
            logits = modules.batch_word_classifier(box_V_ft, map_V)
            self.mid_result['{}_logits'.format(key)] = logits

            with tf.name_scope('{}_classification_loss'.format(key)):
                gt = self.batch['{}_selection_gt'.format(key)]
                num_used_box = self.batch['{}_num_used_box'.format(key)]
                used_mask = tf.sequence_mask(num_used_box, maxlen=num_b,
                                             dtype=tf.float32)
                self.mid_result['{}_used_mask'.format(key)] = used_mask

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
        desc_box_V_ft = tf.reshape(desc_box_V_ft_flat, [-1, num_desc_box, V_DIM])
        self.mid_result['region_visbox'] = tf.reshape(tf.gather(
            visbox_flat, tf.reshape(desc_box_idx, [-1])), [-1, num_desc_box, 4])

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
            tf.reshape(self.batch['lr_desc_idx'], [-1, num_desc_box * lr_num_k]),
            offset=num_desc_box)
        lr_map_V_flat = tf.gather(desc_map_V_flat,
                                  tf.reshape(lr_desc_idx_flat, [-1]))
        lr_map_V = tf.reshape(lr_map_V_flat, [-1, num_desc_box, lr_num_k, V_DIM])
        lr_gt = self.batch['lr_gt']

        if self.num_aug_retrieval > 0:
            lr_map_V, lr_gt = self.aug_retrieval(
                lr_map_V, lr_gt, self.num_aug_retrieval, num_desc_box,
                lr_num_k, V_DIM, scope='aug_LR')

        # Language Retrieval Classifier
        lr_logits = modules.batch_word_classifier(desc_box_V_ft, lr_map_V)
        self.mid_result['lr_logits'] = lr_logits
        self.mid_result['aug_lr_gt'] = lr_gt

        with tf.name_scope('LR_classification_loss'):
            num_used_desc = self.batch['num_used_desc']
            used_desc_mask = tf.sequence_mask(
                num_used_desc, maxlen=num_desc_box, dtype=tf.float32)
            self.mid_result['lr_used_desc_mask'] = used_desc_mask

            loss, acc, top_k_acc = self.n_way_classification_loss(
                lr_logits, lr_gt, used_desc_mask)
            self.losses['retrieval_L'] = loss
            self.report['retrieval_L_loss'] = loss
            self.report['retrieval_L_acc'] = acc
            self.report['retrieval_L_top_k_acc'] = top_k_acc

        # Image retrieval - for each description - classification over images
        ir_num_k = self.data_cfg.ir_num_k
        ir_box_idx_flat = modules.batch_box(
            tf.reshape(self.batch['ir_box_idx'], [-1, num_desc_box * ir_num_k]),
            offset=num_desc_box)
        ir_box_V_ft_flat = tf.gather(V_ft_flat,
                                     tf.reshape(ir_box_idx_flat, [-1]))
        ir_box_V = tf.reshape(ir_box_V_ft_flat,
                              [-1, num_desc_box, ir_num_k, V_DIM])
        ir_gt = self.batch['ir_gt']
        self.mid_result['retrieval_I_visbox'] = tf.reshape(tf.gather(
            visbox_flat, tf.reshape(ir_box_idx_flat, [-1])),
            [-1, num_desc_box, ir_num_k, 4])

        if self.num_aug_retrieval > 0:
            ir_box_V, ir_gt = self.aug_retrieval(
                ir_box_V, ir_gt, self.num_aug_retrieval, num_desc_box,
                ir_num_k, V_DIM, scope='aug_IR')

        # Image Retrieval Classifier
        ir_logits = modules.batch_word_classifier(desc_map_V, ir_box_V)
        self.mid_result['ir_logits'] = ir_logits
        self.mid_result['aug_ir_gt'] = ir_gt

        with tf.name_scope('IR_classification_loss'):
            num_used_desc = self.batch['num_used_desc']
            used_desc_mask = tf.sequence_mask(
                num_used_desc, maxlen=num_desc_box, dtype=tf.float32)
            self.mid_result['ir_used_desc_mask'] = used_desc_mask

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
        if self.description_task == 'blank-fill':
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

        logits_flat, pred_flat, pred_len_flat = modules.decode_L(
            in_L_flat, self.decoder_dim, self.glove_map,
            self.vocab['dict']['<s>'], unroll_type='teacher_forcing',
            seq=desc_flat, seq_len=desc_len_flat + 1,
            output_layer=self.word_predictor, is_train=is_train)
        self.mid_result['pred'] = tf.reshape(
            pred_flat, [-1, num_desc_box, tf.shape(pred_flat)[-1]])
        self.mid_result['pred_len'] = tf.reshape(
            pred_len_flat, [-1, num_desc_box])

        _, greedy_flat, greedy_len_flat = modules.decode_L(
            in_L_flat, self.decoder_dim, self.glove_map,
            self.vocab['dict']['<s>'], unroll_type='greedy',
            end_token=self.vocab['dict']['<e>'],
            max_seq_len=self.data_cfg.max_len['region'] + 1,
            output_layer=self.word_predictor, is_train=is_train)
        self.mid_result['greedy'] = tf.reshape(
            greedy_flat, [-1, num_desc_box, tf.shape(greedy_flat)[-1]])
        self.mid_result['greedy_len'] = tf.reshape(
            greedy_len_flat, [-1, num_desc_box])

        with tf.name_scope('description_loss'):
            desc_used_mask = tf.sequence_mask(
                num_used_desc, maxlen=num_desc_box, dtype=tf.float32)
            self.mid_result['desc_used_mask'] = desc_used_mask
            loss = self.flat_description_loss(
                logits_flat, desc_flat, desc_len_flat + 1, desc_used_mask)
            pred_token_acc, pred_seq_acc = self.flat_description_accuracy(
                pred_flat, pred_len_flat, desc_flat, desc_len_flat + 1,
                desc_used_mask)
            greedy_token_acc, greedy_seq_acc = self.flat_description_accuracy(
                greedy_flat, greedy_len_flat, desc_flat, desc_len_flat + 1,
                desc_used_mask)

            self.losses['description'] = loss
            self.report['description_loss'] = loss
            self.report['pred_token_acc'] = pred_token_acc
            self.report['pred_seq_acc'] = pred_seq_acc
            self.report['greedy_token_acc'] = greedy_token_acc
            self.report['greedy_seq_acc'] = greedy_seq_acc

        with tf.name_scope('prepare_summary'):
            for key in self.target_entry:
                if key == 'attribute':
                    self.vis_image['{}_classification'.format(key)] =\
                        self.vis_binary_image_classification(
                            self.batch['image'],
                            self.mid_result['{}_visbox'.format(key)],
                            self.mid_result['{}_logits'.format(key)],
                            self.batch['{}_selection_gt'.format(key)],
                            self.batch['{}_candidate'.format(key)],
                            self.batch['{}_candidate_len'.format(key)],
                            self.batch['{}_candidate_name'.format(key)],
                            self.mid_result['{}_used_mask'.format(key)],
                            vis_numbox=VIS_NUMBOX, line_width=LINE_WIDTH)
                else:
                    self.vis_image['{}_classification'.format(key)] =\
                        self.vis_n_way_image_classification(
                            self.batch['image'],
                            self.mid_result['{}_visbox'.format(key)],
                            self.mid_result['{}_logits'.format(key)],
                            self.batch['{}_selection_gt'.format(key)],
                            self.batch['{}_candidate'.format(key)],
                            self.batch['{}_candidate_len'.format(key)],
                            self.batch['{}_candidate_name'.format(key)],
                            self.mid_result['{}_used_mask'.format(key)],
                            vis_numbox=VIS_NUMBOX, line_width=LINE_WIDTH)

            self.vis_image['retrieval_L'] = self.vis_retrieval_L(
                self.batch['image'],
                self.mid_result['region_visbox'],
                self.batch['desc'],
                self.batch['desc_len'],
                self.batch['lr_desc_idx'],
                self.data_cfg.lr_num_k,
                self.mid_result['lr_logits'],
                self.mid_result['aug_lr_gt'],
                self.mid_result['lr_used_desc_mask'],
                vis_numbox=VIS_NUMBOX, line_width=LINE_WIDTH)

            self.vis_image['retrieval_I'] = self.vis_retrieval_I(
                self.batch['image'],
                self.mid_result['retrieval_I_visbox'],
                self.batch['desc'],
                self.batch['desc_len'],
                self.num_aug_retrieval,
                self.data_cfg.ir_num_k,
                self.mid_result['ir_logits'],
                self.mid_result['aug_ir_gt'],
                self.mid_result['ir_used_desc_mask'],
                vis_numbox=VIS_NUMBOX, line_width=LINE_WIDTH)

            self.vis_image['description'] = self.vis_description(
                self.batch['image'],
                self.mid_result['region_visbox'],
                self.batch['desc'],
                self.batch['desc_len'],
                self.batch['blank_desc'],
                self.batch['blank_desc_len'],
                self.description_task == 'blank-fill',
                self.mid_result['pred'],
                self.mid_result['pred_len'],
                self.mid_result['greedy'],
                self.mid_result['greedy_len'],
                self.mid_result['desc_used_mask'],
                vis_numbox=VIS_NUMBOX, line_width=LINE_WIDTH)
        # loss
        self.v_loss = 0
        self.l_loss = 0

        for key in self.target_entry:
            self.v_loss += self.losses[key]
        self.v_loss += self.losses['retrieval_L']
        self.v_loss += self.losses['retrieval_I']

        self.l_loss += self.losses['description']

        self.loss = self.v_loss + self.l_loss

        self.report['total_loss'] = self.loss
        self.report['total_v_loss'] = self.v_loss
        self.report['total_l_loss'] = self.l_loss

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
