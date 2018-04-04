import collections
import cPickle
import h5py
import os
import numpy as np
import tensorflow as tf

from util import log

NUM_CONFIG = {
    'obj_pred': 5,
    'obj_att': 4,
    'attr_pred': 1,
    'attr_att': 1,
    'attr_blank_fill': 5,
    'obj_blank_fill': 5,
    'caption_att': 5,
}

RANDOM_STATE = np.random.RandomState(123)


class Dataset(object):
    def __init__(self, data_dir, split, name='vlmap_memft'):
        self.name = name
        self.split = split

        log.warn('loading image_info ..')
        image_info_path = os.path.join(
            data_dir, '{}_image_info.pkl'.format(split))
        image_info = cPickle.load(open(image_info_path, 'rb'))
        self._ids = image_info['image_ids']
        self.image_id2idx = image_info['image_id2idx']
        log.info('loading image_info done')

        log.warn('loading processed data ..')
        processed_path = os.path.join(
            data_dir, '{}_processed.pkl'.format(split))
        self.processed = cPickle.load(open(processed_path, 'rb'))
        log.info('loading processed done')

        log.warn('loading answer_dict ..')
        answer_dict_path = os.path.join(data_dir, 'answer_dict.pkl')
        self.answer_dict = cPickle.load(open(answer_dict_path, 'rb'))
        self.num_answers = len(self.answer_dict['vocab'])
        log.info('loading answer_dict done')

        with h5py.File(os.path.join(
            data_dir, '{}_vfeat.hdf5'.format(split)), 'r') as f:

            self.vfeat_dim = int(f['data_info']['vfeat_dim'].value)
            self.max_box_num = int(f['data_info']['max_box_num'].value)
            log.warn('loading {} image_features ..'.format(split))
            self.image_features = np.array(f.get('image_features'))
            log.warn('loading {} normal_boxes ..'.format(split))
            self.normal_boxes = np.array(f.get('normal_boxes'))
            log.warn('loading {} num_boxes ..'.format(split))
            self.num_boxes = np.array(f.get('num_boxes'))
            log.warn('loading {} spatial_features ..'.format(split))
            self.spatial_features = np.array(f.get('spatial_features'))
            log.warn('loading {} features done ..'.format(split))

        log.info('dataset {} {} init done'.format(name, split))

    def get_config(self):
        config = collections.namedtuple('dataset_config', [])
        return config

    def get_data(self, image_id):
        image_idx = self.image_id2idx[image_id]
        image_ft = self.image_features[image_idx]
        spatial_ft = self.spatial_features[image_idx]
        normal_box = self.normal_boxes[image_idx]
        num_box = self.num_boxes[image_idx]

        ret = {
            'image_id': np.array(image_id, dtype=np.int32),
            'image_ft': image_ft,
            'spatial_ft': spatial_ft,
            'normal_boxes': normal_box,
            'num_boxes': num_box
        }

        entry = self.processed[image_id]
        """
        object_predict
        """
        idx_list = list(range(len(entry['object_predict'])))
        RANDOM_STATE.shuffle(idx_list)
        idx_list = idx_list[:NUM_CONFIG['obj_pred']]
        num_valid_data = len(idx_list)
        while len(idx_list) < NUM_CONFIG['obj_pred']:
            idx_list.append(idx_list[-1])
        labels, weights, normal_boxes = [], [], []
        for idx in idx_list:
            e = entry['object_predict'][idx]
            weight = np.zeros([self.max_box_num], dtype=np.float32)
            weight[e['p_idx']] = e['p_weight']
            weights.append(weight)
            labels.append(e['label'])
            normal_boxes.append(e['normal_box'])
        ret.update({
            'obj_pred/num': np.array(num_valid_data, dtype=np.int32),
            'obj_pred/labels': np.array(labels, dtype=np.int32),
            'obj_pred/weights': np.array(weights, dtype=np.float32),
            'obj_pred/normal_boxes': np.array(
                normal_boxes, dtype=np.float32),
        })
        """
        attribute_predict
        """
        idx_list = list(range(len(entry['attr_predict'])))
        RANDOM_STATE.shuffle(idx_list)
        idx_list = idx_list[:NUM_CONFIG['attr_pred']]
        num_valid_data = len(idx_list)
        while len(idx_list) < NUM_CONFIG['attr_pred']:
            idx_list.append(idx_list[-1])
        object_labels, weights, normal_boxes = [], [], []
        labels = np.zeros([len(idx_list), self.num_answers],
                          dtype=np.float32)
        for i, idx in enumerate(idx_list):
            e = entry['attr_predict'][idx]
            weight = np.zeros([self.max_box_num], dtype=np.float32)
            weight[e['p_idx']] = e['p_weight']
            weights.append(weight)
            object_labels.append(e['object_label'])
            labels[i, e['labels']] = 1.0
            normal_boxes.append(e['normal_box'])
        ret.update({
            'attr_pred/num': np.array(num_valid_data, dtype=np.int32),
            'attr_pred/labels': labels,
            'attr_pred/object_labels': np.array(
                object_labels, dtype=np.int32),
            'attr_pred/weights': np.array(weights, dtype=np.float32),
            'attr_pred/normal_boxes': np.array(
                normal_boxes, dtype=np.float32),
        })
        """
        object_attend
        """
        idx_list = list(range(len(entry['object_attend'])))
        RANDOM_STATE.shuffle(idx_list)
        idx_list = idx_list[:NUM_CONFIG['obj_att']]
        num_valid_data = len(idx_list)
        while len(idx_list) < NUM_CONFIG['obj_att']:
            idx_list.append(idx_list[-1])
        maxlen = max([len(entry['object_attend'][idx]['w_tokens'])
                        for idx in idx_list])
        att_scores, word_tokens_len = [], []
        word_tokens = np.zeros([len(idx_list), maxlen], dtype=np.int32)
        for i, idx in enumerate(idx_list):
            e = entry['object_attend'][idx]
            att_score = np.zeros([self.max_box_num], dtype=np.float32)
            att_score[e['s_idx']] = e['s_value']
            att_scores.append(att_score)
            word_tokens_len.append(len(e['w_tokens']))
            word_tokens[i, :word_tokens_len[i]] = e['w_tokens']
        ret.update({
            'obj_att/num': np.array(num_valid_data, dtype=np.int32),
            'obj_att/att_scores': np.array(att_scores, dtype=np.float32),
            'obj_att/word_tokens': word_tokens,
            'obj_att/word_tokens_len': np.array(
                word_tokens_len, dtype=np.int32),
        })
        """
        attribute_attend
        """
        idx_list = list(range(len(entry['attr_attend'])))
        RANDOM_STATE.shuffle(idx_list)
        idx_list = idx_list[:NUM_CONFIG['attr_att']]
        num_valid_data = len(idx_list)
        while len(idx_list) < NUM_CONFIG['attr_att']:
            idx_list.append(idx_list[-1])
        maxlen = max([len(entry['attr_attend'][idx]['w_tokens'])
                        for idx in idx_list])
        att_scores, word_tokens_len = [], []
        word_tokens = np.zeros([len(idx_list), maxlen], dtype=np.int32)
        for i, idx in enumerate(idx_list):
            e = entry['attr_attend'][idx]
            att_score = np.zeros([self.max_box_num], dtype=np.float32)
            att_score[e['s_idx']] = e['s_value']
            att_scores.append(att_score)
            word_tokens_len.append(len(e['w_tokens']))
            word_tokens[i, :word_tokens_len[i]] = e['w_tokens']
        ret.update({
            'attr_att/num': np.array(num_valid_data, dtype=np.int32),
            'attr_att/att_scores': np.array(att_scores, dtype=np.float32),
            'attr_att/word_tokens': word_tokens,
            'attr_att/word_tokens_len': np.array(
                word_tokens_len, dtype=np.int32),
        })
        """
        object_blank_fill
        """
        idx_list = list(range(len(entry['obj_blank_fill'])))
        RANDOM_STATE.shuffle(idx_list)
        idx_list = idx_list[:NUM_CONFIG['obj_blank_fill']]
        num_valid_data = len(idx_list)
        while len(idx_list) < NUM_CONFIG['obj_blank_fill']:
            idx_list.append(idx_list[-1])
        maxlen = max([len(entry['obj_blank_fill'][idx]['blank'])
                        for idx in idx_list])
        weights, normal_boxes, fills, blanks_len = [], [], [], []
        blanks = np.zeros([len(idx_list), maxlen], dtype=np.int32)
        for i, idx in enumerate(idx_list):
            e = entry['obj_blank_fill'][idx]
            weight = np.zeros([self.max_box_num], dtype=np.float32)
            weight[e['p_idx']] = e['p_weight']
            weights.append(weight)
            normal_boxes.append(e['normal_box'])
            blanks_len.append(len(e['blank']))
            blanks[i, :blanks_len[i]] = e['blank']
            fills.append(e['fill'])
        ret.update({
            'obj_blank_fill/num': np.array(num_valid_data, dtype=np.int32),
            'obj_blank_fill/weights': np.array(weights, dtype=np.float32),
            'obj_blank_fill/normal_boxes': np.array(
                normal_boxes, dtype=np.float32),
            'obj_blank_fill/fills': np.array(fills, dtype=np.int32),
            'obj_blank_fill/blanks': blanks,
            'obj_blank_fill/blanks_len': np.array(blanks_len, dtype=np.int32),
        })
        """
        attribute_blank_fill
        """
        idx_list = list(range(len(entry['attr_blank_fill'])))
        RANDOM_STATE.shuffle(idx_list)
        idx_list = idx_list[:NUM_CONFIG['attr_blank_fill']]
        num_valid_data = len(idx_list)
        while len(idx_list) < NUM_CONFIG['attr_blank_fill']:
            idx_list.append(idx_list[-1])
        maxlen = max([len(entry['attr_blank_fill'][idx]['blank'])
                        for idx in idx_list])
        weights, normal_boxes, fills, blanks_len = [], [], [], []
        blanks = np.zeros([len(idx_list), maxlen], dtype=np.int32)
        for i, idx in enumerate(idx_list):
            e = entry['attr_blank_fill'][idx]
            weight = np.zeros([self.max_box_num], dtype=np.float32)
            weight[e['p_idx']] = e['p_weight']
            weights.append(weight)
            normal_boxes.append(e['normal_box'])
            blanks_len.append(len(e['blank']))
            blanks[i, :blanks_len[i]] = e['blank']
            fills.append(e['fill'])
        ret.update({
            'attr_blank_fill/num': np.array(num_valid_data, dtype=np.int32),
            'attr_blank_fill/weights': np.array(weights, dtype=np.float32),
            'attr_blank_fill/normal_boxes': np.array(
                normal_boxes, dtype=np.float32),
            'attr_blank_fill/fills': np.array(fills, dtype=np.int32),
            'attr_blank_fill/blanks': blanks,
            'attr_blank_fill/blanks_len': np.array(blanks_len, dtype=np.int32),
        })
        """
        caption_attend
        """
        idx_list = list(range(len(entry['caption_attend'])))
        RANDOM_STATE.shuffle(idx_list)
        idx_list = idx_list[:NUM_CONFIG['caption_att']]
        num_valid_data = len(idx_list)
        while len(idx_list) < NUM_CONFIG['caption_att']:
            idx_list.append(idx_list[-1])
        maxlen = max([len(entry['caption_attend'][idx]['w_tokens'])
                        for idx in idx_list])
        att_scores, word_tokens_len = [], []
        word_tokens = np.zeros([len(idx_list), maxlen], dtype=np.int32)
        for i, idx in enumerate(idx_list):
            e = entry['caption_attend'][idx]
            att_score = np.zeros([self.max_box_num], dtype=np.float32)
            att_score[e['s_idx']] = e['s_value']
            att_scores.append(att_score)
            word_tokens_len.append(len(e['w_tokens']))
            word_tokens[i, :word_tokens_len[i]] = e['w_tokens']
        ret.update({
            'cap_att/num': np.array(num_valid_data, dtype=np.int32),
            'cap_att/att_scores': np.array(att_scores, dtype=np.float32),
            'cap_att/word_tokens': word_tokens,
            'cap_att/word_tokens_len': np.array(
                word_tokens_len, dtype=np.int32),
        })
        return ret

    def get_shapes(self):
        ret_shapes = {
            'image_id': (),
            'image_ft': [self.max_box_num, self.vfeat_dim],
            'spatial_ft': [self.max_box_num, 6],
            'normal_boxes': [self.max_box_num, 4],
            'num_boxes': (),
            'obj_pred/num': (),
            'obj_pred/labels': [NUM_CONFIG['obj_pred']],
            'obj_pred/weights': [NUM_CONFIG['obj_pred'], self.max_box_num],
            'obj_pred/normal_boxes': [NUM_CONFIG['obj_pred'], 4],
            'attr_pred/num': (),
            'attr_pred/labels': [NUM_CONFIG['attr_pred'], self.num_answers],
            'attr_pred/object_labels': [NUM_CONFIG['attr_pred']],
            'attr_pred/weights': [NUM_CONFIG['attr_pred'], self.max_box_num],
            'attr_pred/normal_boxes': [NUM_CONFIG['attr_pred'], 4],
            'obj_att/num': (),
            'obj_att/att_scores': [NUM_CONFIG['obj_att'], self.max_box_num],
            'obj_att/word_tokens': [NUM_CONFIG['obj_att'], None],
            'obj_att/word_tokens_len': [NUM_CONFIG['obj_att']],
            'attr_att/num': (),
            'attr_att/att_scores': [NUM_CONFIG['attr_att'], self.max_box_num],
            'attr_att/word_tokens': [NUM_CONFIG['obj_att'], None],
            'attr_att/word_tokens_len': [NUM_CONFIG['obj_att']],
            'obj_blank_fill/num': (),
            'obj_blank_fill/weights': [NUM_CONFIG['obj_blank_fill'], self.max_box_num],
            'obj_blank_fill/normal_boxes': [NUM_CONFIG['obj_blank_fill'], 4],
            'obj_blank_fill/fills': [NUM_CONFIG['obj_blank_fill']],
            'obj_blank_fill/blanks': [NUM_CONFIG['obj_blank_fill'], None],
            'obj_blank_fill/blanks_len': [NUM_CONFIG['obj_blank_fill']],
            'attr_blank_fill/num': (),
            'attr_blank_fill/weights': [NUM_CONFIG['attr_blank_fill'], self.max_box_num],
            'attr_blank_fill/normal_boxes': [NUM_CONFIG['attr_blank_fill'], 4],
            'attr_blank_fill/fills': [NUM_CONFIG['attr_blank_fill']],
            'attr_blank_fill/blanks': [NUM_CONFIG['attr_blank_fill'], None],
            'attr_blank_fill/blanks_len': [NUM_CONFIG['attr_blank_fill']],
            'cap_att/num': (),
            'cap_att/att_scores': [NUM_CONFIG['caption_att'], self.max_box_num],
            'cap_att/word_tokens': [NUM_CONFIG['caption_att'], None],
            'cap_att/word_tokens_len': [NUM_CONFIG['caption_att']],
        }
        return ret_shapes

    def get_types(self):
        ret_type = {
            'image_id': tf.int32,
            'image_ft': tf.float32,
            'spatial_ft': tf.float32,
            'normal_boxes': tf.float32,
            'num_boxes': tf.int32,
            'obj_pred/num': tf.int32,
            'obj_pred/labels': tf.int32,
            'obj_pred/weights': tf.float32,
            'obj_pred/normal_boxes': tf.float32,
            'attr_pred/num': tf.int32,
            'attr_pred/labels': tf.float32,
            'attr_pred/object_labels': tf.int32,
            'attr_pred/weights': tf.float32,
            'attr_pred/normal_boxes': tf.float32,
            'obj_att/num': tf.int32,
            'obj_att/att_scores': tf.float32,
            'obj_att/word_tokens': tf.int32,
            'obj_att/word_tokens_len': tf.int32,
            'attr_att/num': tf.int32,
            'attr_att/att_scores': tf.float32,
            'attr_att/word_tokens': tf.int32,
            'attr_att/word_tokens_len': tf.int32,
            'obj_blank_fill/num': tf.int32,
            'obj_blank_fill/weights': tf.float32,
            'obj_blank_fill/normal_boxes': tf.float32,
            'obj_blank_fill/fills': tf.int32,
            'obj_blank_fill/blanks': tf.int32,
            'obj_blank_fill/blanks_len': tf.int32,
            'attr_blank_fill/num': tf.int32,
            'attr_blank_fill/weights': tf.float32,
            'attr_blank_fill/normal_boxes': tf.float32,
            'attr_blank_fill/fills': tf.int32,
            'attr_blank_fill/blanks': tf.int32,
            'attr_blank_fill/blanks_len': tf.int32,
            'cap_att/num': tf.int32,
            'cap_att/att_scores': tf.float32,
            'cap_att/word_tokens': tf.int32,
            'cap_att/word_tokens_len': tf.int32,
        }
        return ret_type

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset ({} [{}], {} examples)'.format(
            self.name, self.split, len(self))


def create_ops(batch_size,
               dataset,
               split,
               is_train=True,
               scope='vlmap_memft',
               shuffle=True):

    with tf.device('/cpu:0'), tf.name_scope(scope):
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(dataset.ids))

        if is_train and shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=3000)

        def load_fn(image_id):
            ret = dataset.get_data(image_id)
            ret_list = [ret[key] for key in sorted(ret.keys())]
            return ret_list

        def load_pyfunc(image_id):
            ret_type = dataset.get_types()
            ret_type_list = [ret_type[key] for key in sorted(ret_type.keys())]
            pyfunc_ret_list = tf.py_func(load_fn, inp=[image_id],
                                         Tout=ret_type_list, name='input_pyfunc')
            pyfunc_ret = {key: val for key, val in
                          zip(sorted(ret_type.keys()), pyfunc_ret_list)}
            shapes = dataset.get_shapes()
            for key, shape in shapes.items():
                pyfunc_ret[key].set_shape(shape)
            return pyfunc_ret

        tf_dataset = tf_dataset.map(load_pyfunc)

    tf_dataset = tf_dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=dataset.get_shapes()
    )

    tf_dataset = tf_dataset.prefetch(10)

    if is_train:
        tf_dataset = tf_dataset.repeat(1000)  # repeat 1000 epoch

    iterator = tf_dataset.make_one_shot_iterator()
    batch_ops = iterator.get_next()

    return batch_ops
