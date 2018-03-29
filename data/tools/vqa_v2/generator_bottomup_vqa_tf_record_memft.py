import argparse
import cPickle
import h5py
import json
import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from util import log, tf_util

NUM_BOXES = 36
FEATURE_DIM = 2048

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bottomup_vqa_data_dir', type=str,
                    default='data/VQA_v2/bottomup_vqa_data', help=' ')
parser.add_argument('--save_dir', type=str,
                    default='data/preprocessed/vqa_v2'
                    '/bottomup_vqa_tf_record_memft', help=' ')
parser.add_argument('--use_train_reserve', action='store_true', default=False)
parser.add_argument('--num_record_per_shard', type=int, default=1024, help=' ')
config = parser.parse_args()

config.data_info_path = os.path.join(config.save_dir, 'data_info.hdf5')
config.vfeat_path = os.path.join(config.save_dir, 'vfeat_bottomup_36.hdf5')

if not os.path.exists(config.save_dir):
    log.warn('create directory: {}'.format(config.save_dir))
    os.makedirs(config.save_dir)
else:
    raise ValueError('Do not overwrite: {}'.format(config.save_dir))

ans2label_path = os.path.join(
    config.bottomup_vqa_data_dir, 'cache', 'trainval_ans2label.pkl')
label2ans_path = os.path.join(
    config.bottomup_vqa_data_dir, 'cache', 'trainval_label2ans.pkl')

ans2label = cPickle.load(open(ans2label_path, 'rb'))
label2ans = cPickle.load(open(label2ans_path, 'rb'))
num_ans_candidates = len(ans2label)

dict_path = os.path.join(config.bottomup_vqa_data_dir, 'dictionary.pkl')
word2idx, idx2word = cPickle.load(open(dict_path, 'rb'))
vocab = {
    'vocab': idx2word,
    'dict': word2idx
}

def tokenize_question(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
    words = sentence.split()
    tokens = []
    for w in words:
        tokens.append(vocab['dict'][w])
    return tokens

img_id2idx = {
    'train': cPickle.load(open(
        os.path.join(config.bottomup_vqa_data_dir, 'train36_imgid2idx.pkl'))),
    'val': cPickle.load(open(
        os.path.join(config.bottomup_vqa_data_dir, 'val36_imgid2idx.pkl'))),
}

h5_file = {
    'train': h5py.File(
        os.path.join(config.bottomup_vqa_data_dir, 'train36.hdf5'), 'r'),
    'val': h5py.File(
        os.path.join(config.bottomup_vqa_data_dir, 'val36.hdf5'), 'r'),
}
features_split = {}
log.infov('loading train features ...')
features_split['train'] = np.array(h5_file['train'].get('image_features'))
log.infov('loading val features ...')
features_split['val'] = np.array(h5_file['val'].get('image_features'))
log.warn('loading is done')

spatials_split = {
    'train': np.array(h5_file['train'].get('spatial_features')),
    'val': np.array(h5_file['val'].get('spatial_features')),
}
num_feat = {
    'train': features_split['train'].shape[0],
    'val': features_split['val'].shape[0]
}
num_feat['total'] = num_feat['train'] + num_feat['val']

# use number of training features as an offeset
for img_id in img_id2idx['val']:
    img_id2idx['val'][img_id] = \
        img_id2idx['val'][img_id] + num_feat['train']

vfeat_h5 = h5py.File(config.vfeat_path, 'w')

features = vfeat_h5.create_dataset(
    'image_features', (num_feat['total'], NUM_BOXES, FEATURE_DIM), 'f')
log.infov('writing train features to h5_file...')
features[:num_feat['train'], :, :] = features_split['train']
log.infov('writing val features to h5_file...')
features[num_feat['train']:, :, :] = features_split['val']
log.warn('writing features done')

# spatials: [num_feat, 36, 6] (normalized: x1, y1, x2, y2, w, h)
spatials = np.concatenate([spatials_split['train'],
                           spatials_split['val']], axis=0)
vfeat_h5['spatial_features'] = spatials

# normalized boxes coordinates
x1 = spatials[:, :, 0]
y1 = spatials[:, :, 1]
x2 = spatials[:, :, 2]
y2 = spatials[:, :, 3]

normal_boxes = np.stack([x1, y1, x2, y2], axis=-1)
vfeat_h5['normal_boxes'] = normal_boxes

num_boxes = np.zeros([num_feat['total']], dtype=np.int32) + NUM_BOXES
vfeat_h5['num_boxes'] = num_boxes

vfeat_data_info = vfeat_h5.create_group('data_info')
vfeat_data_info['vfeat_dim'] = FEATURE_DIM
vfeat_data_info['max_box_num'] = NUM_BOXES
vfeat_data_info['pretrained_param_path'] = 'bottom_up_attention_36'
vfeat_h5.close()

h5_file['train'].close()
h5_file['val'].close()

"""
Following functions are copied from the following repository:
    - https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_num': question['image_num'],
        'image_id': question['image_id'],
        'image_path': question['image_path'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    assert name in ['train', 'val']
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        assert_eq(question['question_id'], answer['question_id'])
        assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        question['image_num'] = question['image_id']
        question['image_path'] = '{}2014/COCO_{}2014_{:012d}.jpg'.format(
            name, name, img_id)
        question['image_id'] = question['image_path'].replace('/', '-')
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries

max_q_len = 0
max_num_answer = 0
num_data = {}
for split in ['train', 'val']:
    entries = _load_dataset(config.bottomup_vqa_data_dir, split, img_id2idx[split])
    max_length = 14  # this number comes from bottom-up-attention-vqa
    for entry in tqdm(entries, desc='tokenize {}'.format(split)):
        tokens = tokenize_question(entry['question'])
        tokens = tokens[:max_length]
        entry['q_token'] = tokens

    tf_record_dir = os.path.join(config.save_dir, split)
    if not os.path.exists(tf_record_dir):
        log.warn('create directory: {}'.format(tf_record_dir))
        os.makedirs(tf_record_dir)

    num_data[split] = len(entries)
    num_shards = len(entries) / config.num_record_per_shard + 1
    for i, entry in enumerate(tqdm(entries,
                                   desc='write tfrecord {}'.format(split))):
        if i % config.num_record_per_shard == 0:
            shard_id = int(i / config.num_record_per_shard)
            shard_name = '{}-{:05d}-of-{:05d}'.format(
                split, shard_id, num_shards)
            shard_path = os.path.join(config.save_dir, split, shard_name)
            if os.path.exists(shard_path):
                raise ValueError('Existing shard path: {}'.format(shard_path))
            tf_record_writer = tf.python_io.TFRecordWriter(shard_path)

        max_q_len = max(max_q_len, len(entry['q_token']))
        max_num_answer = max(max_num_answer, len(entry['answer']['labels']))
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'qid': tf_util.int64_feature(int(entry['question_id'])),
            'image_id': tf_util.bytes_feature(str(entry['image_id'])),
            'image_idx': tf_util.int64_feature(int(entry['image'])),
            'q_intseq/list': tf_util.int64_feature(entry['q_token']),
            'q_intseq/len': tf_util.int64_feature(len(entry['q_token'])),
            'answers/ids': tf_util.int64_feature(entry['answer']['labels']),
            'answers/scores': tf_util.float_feature(entry['answer']['scores']),
        }))
        tf_record_writer.write(tf_example.SerializeToString())

# Construct image_info
used_image_paths = set()
image_id2idx = {}
image_path2idx = {}
image_num2path = {}
for split in ['train', 'val']:
    for image_num, image_idx in img_id2idx[split].items():
        image_path = '{}2014/COCO_{}2014_{:012d}.jpg'.format(
            split, split, image_num)
        image_id = image_path.replace('/', '-')
        used_image_paths.add(image_path)
        image_id2idx[image_id] = image_idx
        image_path2idx[image_path] = image_idx
        image_num2path[image_num] = image_path
used_image_paths = list(used_image_paths)
image_info = {
    'used_image_paths': used_image_paths,
    'image_id2idx': image_id2idx,
    'image_path2idx': image_path2idx,
    'image_num2path': image_num2path,
}
image_info_path = os.path.join(config.save_dir, 'image_info.json')
json.dump(image_info, open(image_info_path, 'w'))
log.warn('used image information is saved in: {}'.format(image_info_path))


# save vocab and answer dict
vocab_path = os.path.join(config.save_dir, 'vocab.json')
json.dump(vocab, open(vocab_path, 'w'))
log.warn('vocab is saved in: {}'.format(vocab_path))

answer_dict = {
    'dict': ans2label,
    'vocab': label2ans,
}
answer_dict_path = os.path.join(config.save_dir, 'answer_dict.json')
json.dump(answer_dict, open(answer_dict_path, 'w'))
log.warn('answer_dict is saved in: {}'.format(answer_dict_path))

data_info_h5 = h5py.File(config.data_info_path, 'w')
data_info = data_info_h5.create_group('data_info')
data_info['num_answers'] = len(label2ans)
data_info['num_train'] = num_data['train']
data_info['num_val'] = num_data['val']
data_info['max_q_len'] = max_q_len
data_info['max_num_answer'] = max_num_answer
data_info_h5.close()
log.warn('data_info is saved in: {}'.format(config.data_info_path))
