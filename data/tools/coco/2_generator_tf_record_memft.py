import argparse
import h5py
import json
import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from coco_utils import *
from util import log, tf_util

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--caption_split_dir', type=str,
                    default='data/preprocessed/COCO/standrad', help=' ')
parser.add_argument('--caption_dic_path', type=str,
                    default='data/COCO/dic_coco.json', help=' ')
parser.add_argument('--caption_cap_path', type=str,
                    default='data/COCO/cap_coco.json', help=' ')
parser.add_argument('--reference_vcaption_dir', type=str,
                    default='data/preprocessed/vcaption_v2/'
                    'caption_split_objattr_answer_genome_memft_check_all_answer_thres1_50000_thres2_-1')
parser.add_argument('--use_train_reserve', action='store_true', default=False)
parser.add_argument('--num_record_per_shard', type=int, default=1024, help=' ')
config = parser.parse_args()

config.data_dir = os.path.join(config.caption_split_dir, 'tf_record_memft')
config.data_path = os.path.join(config.data_dir, 'data_info.hdf5')
config.id_path = os.path.join(config.data_dir, 'id.txt')

#if not os.path.exists(config.data_dir):
if True:
    log.warn('create directory: {}'.format(config.data_dir))
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)
else:
    raise ValueError(
        'The directory {} already exists. Do not overwrite.'.format(
            config.data_dir))

caption_split_path = os.path.join(
    config.caption_split_dir, "caption_split.json")
caption_split = json.load(open(caption_split_path, 'r'))

log.info('loading vocab')
vocab = json.load(open(
    os.path.join(config.caption_split_dir, 'vocab.json'), 'r'))
log.info('loading frequent answers')
freq_ans = json.load(open(
    os.path.join(config.caption_split_dir, 'frequent_answers.json'), 'r'))
freq_ans_set = set(freq_ans)
log.info('loading merged_captions')


idx2caption = json.load(open(os.path.join(
    config.caption_split_dir, 'idx2caption.json'), 'r'))
idx2caption = {int(key):value for key, value in idx2caption.items()}

log.info('loading caption_split')
caption_split = json.load(open(os.path.join(
    config.caption_split_dir, 'caption_split.json'), 'r'))
log.info('loading is done')


caption_dic = json.load(open(config.caption_dic_path))
caption_cap = json.load(open(config.caption_cap_path))

idx2info = {info['id']: info for info in caption_dic['images']}


used_image_paths = open(
    os.path.join(config.caption_split_dir, 'used_image_path.txt')).read().splitlines()
image_id2idx = {
    image_path.replace('/', '-'): i for i, image_path in enumerate(
    used_image_paths)}

image_path2idx = {image_path: i for i, image_path in enumerate(
    used_image_paths)}
image_num2path = {}
for caption in idx2info.values():
    image_num2path[int(caption['id'])] = caption['file_path']

image_info = {
    'used_image_paths': used_image_paths,
    'image_id2idx': image_id2idx,
    'image_path2idx': image_path2idx,
    'image_num2path': image_num2path,
}
image_info_path = os.path.join(config.data_dir, 'image_info.json')
json.dump(image_info, open(image_info_path, 'w'))
log.warn('used image information is saved in: {}'.format(image_info_path))

"""
How to store answers:
    - in ['data_info'], intseq and intseq_len of all answer candidates are saved
    - for each data point, index of answer candidate is saved as a ground truth
How to store questions:
    - max_q_len is saved in ['data_info']
    - q_intseq is saved for each question. (q_intseq_len is not saved because
      they can be easily obtained from q_intseq)
Data statistics:
    - num_train, num_val, num_test_val, num_test
"""
f = h5py.File(config.data_path, 'w')

if config.use_train_reserve:
    caption_split['train'].extend(caption_split['train-reserve'])
    caption_split['val'].extend(caption_split['val-reserve'])

# encode frequent answers
intseq_ans = []
intseq_ans_len = []
for ans in freq_ans:
    intseq_ans.append([vocab['dict'][t] for t in ans.split()])
    intseq_ans_len.append(len(intseq_ans[-1]))

import ipdb; ipdb.set_trace() 
np_intseq_ans_len = np.array(intseq_ans_len, dtype=np.int32)
max_ans_len = np_intseq_ans_len.max()

np_intseq_ans = np.zeros([len(intseq_ans), max_ans_len], dtype=np.int32)
for i, intseq in enumerate(intseq_ans):
    np_intseq_ans[i, :len(intseq)] = np.array(intseq, dtype=np.int32)
ans_list = freq_ans
ans_dict = {ans: i for i, ans in enumerate(freq_ans)}
num_answers = len(ans_list)

# add frequent answer intseqs to data_info
data_info = f.create_group('data_info')
data_info['intseq_ans'] = np_intseq_ans
data_info['intseq_ans_len'] = np_intseq_ans_len
data_info['max_ans_len'] = np.array(max_ans_len, dtype=np.int32)
data_info['num_answers'] = np.array(num_answers, dtype=np.int32)

"""
For training set QA:
    - if answer is not in freq_ans_set, ignore the example
    - if the vocab is not in GloVe, mark it <unk>
For testing set QA:
    - if answer is not in freq_ans_set, mark it N+1-th answer
    - if the vocab is not in GloVe, mark it <unk>
"""
used_qid = {
    'train': [],
    'val': [],
    'testval': [],
    'test': [],
}
for split in ['train', 'val', 'test', 'testval']:
    tf_record_dir = os.path.join(config.data_dir, split)
    if not os.path.exists(tf_record_dir):
        log.warn('create directory: {}'.format(tf_record_dir))
        os.makedirs(tf_record_dir)


def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1

max_num_answer = 0
max_q_len = 0
for split in ['train']:
    num_shards = len(caption_split[split]) / config.num_record_per_shard + 1
    for i, qid in enumerate(tqdm(caption_split[split],
                                 desc='process {} qids'.format(split))):
        if i % config.num_record_per_shard == 0:
            shard_id = int(i / config.num_record_per_shard)
            shard_name = '{}-{:05d}-of-{:05d}'.format(
                split, shard_id, num_shards)
            shard_path = os.path.join(config.data_dir, split, shard_name)
            if os.path.exists(shard_path):
                raise ValueError('Existing shard path: {}'.format(shard_path))
            tf_record_writer = tf.python_io.TFRecordWriter(shard_path)

        caption = qid2caption[str(qid)]

        answer_count = {}
        for answer in caption['processed_answers']:
            answer_count[answer] = answer_count.get(answer, 0) + 1

        # use tf.sparse_to_dense to make these values to score vector
        answer_ids = []
        answer_scores = []
        for answer, count in answer_count.items():
            if answer not in freq_ans_set:
                continue
            ans_id = ans_dict[answer]
            answer_ids.append(ans_id)
            answer_scores.append(get_score(count))
        if len(answer_ids) == 0:
            continue

        max_num_answer = max(max_num_answer, len(answer_ids))

        unk_id = vocab['dict']['<unk>']
        q_intseq = [vocab['dict'].get(t, unk_id) for t in caption['q_tokens']]
        q_intseq = np.array(q_intseq, dtype=np.int32)
        used_qid[split].append(qid)
        max_q_len = max(max_q_len, len(q_intseq))

        image_id = caption['image_path'].replace('/', '-')
        image_idx = image_id2idx[image_id]
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'qid': tf_util.int64_feature(int(qid)),
            'image_id': tf_util.bytes_feature(str(image_id)),
            'image_idx': tf_util.int64_feature(int(image_idx)),
            'q_intseq/list': tf_util.int64_feature(list(q_intseq)),
            'q_intseq/len': tf_util.int64_feature(len(q_intseq)),
            'answers/ids': tf_util.int64_feature(answer_ids),
            'answers/scores': tf_util.float_feature(answer_scores),
        }))
        tf_record_writer.write(tf_example.SerializeToString())

for split in ['val', 'testval', 'test']:
    num_shards = len(caption_split[split]) / config.num_record_per_shard + 1
    for i, qid in enumerate(tqdm(caption_split[split],
                                 desc='process {} qids'.format(split))):
        if i % config.num_record_per_shard == 0:
            shard_id = int(i / config.num_record_per_shard)
            shard_name = '{}-{:05d}-of-{:05d}'.format(
                split, shard_id, num_shards)
            shard_path = os.path.join(config.data_dir, split, shard_name)
            if os.path.exists(shard_path):
                raise ValueError('Existing shard path: {}'.format(shard_path))
            tf_record_writer = tf.python_io.TFRecordWriter(shard_path)

        caption = qid2caption[str(qid)]

        answer_count = {}
        for answer in caption['processed_answers']:
            answer_count[answer] = answer_count.get(answer, 0) + 1

        # use tf.sparse_to_dense to make these values to score vector
        answer_ids = []
        answer_scores = []
        for answer, count in answer_count.items():
            if answer not in freq_ans_set:
                continue
            ans_id = ans_dict[answer]
            answer_ids.append(ans_id)
            answer_scores.append(get_score(count))

        max_num_answer = max(max_num_answer, len(answer_ids))

        unk_id = vocab['dict']['<unk>']
        q_intseq = [vocab['dict'].get(t, unk_id) for t in caption['q_tokens']]
        q_intseq = np.array(q_intseq, dtype=np.int32)
        used_qid[split].append(qid)
        max_q_len = max(max_q_len, len(q_intseq))

        image_id = caption['image_path'].replace('/', '-')
        image_idx = image_id2idx[image_id]

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'qid': tf_util.int64_feature(int(qid)),
            'image_id': tf_util.bytes_feature(str(image_id)),
            'image_idx': tf_util.int64_feature(int(image_idx)),
            'q_intseq/list': tf_util.int64_feature(list(q_intseq)),
            'q_intseq/len': tf_util.int64_feature(len(q_intseq)),
            'answers/ids': tf_util.int64_feature(answer_ids),
            'answers/scores': tf_util.float_feature(answer_scores),
        }))
        tf_record_writer.write(tf_example.SerializeToString())


log.warn('write to data_info')
data_info['num_train'] = len(used_qid['train'])
data_info['num_val'] = len(used_qid['val'])
data_info['num_testval'] = len(used_qid['testval'])
data_info['num_test'] = len(used_qid['test'])
data_info['max_q_len'] = np.array(max_q_len, dtype=np.int32)
data_info['max_num_answer'] = max_num_answer
f.close()

log.warn('write to id file')
fid = open(config.id_path, 'w')
for key in ['train', 'val', 'testval', 'test']:
    for qid in used_qid[key]:
        fid.write(str(qid) + '\n')
fid.close()
log.infov('data file is saved to: {}'.format(config.data_path))
log.infov('id file is saved to: {}'.format(config.id_path))

vocab_path = os.path.join(config.data_dir, 'vocab.json')
json.dump(vocab, open(vocab_path, 'w'))
log.infov('vocab is saved to: {}'.format(vocab_path))

answer_dict_path = os.path.join(config.data_dir, 'answer_dict.json')
answer_dict = {
    'vocab': ans_list,
    'dict': ans_dict
}
json.dump(answer_dict, open(answer_dict_path, 'w'))
log.infov('answer_dict is saved to: {}'.format(answer_dict_path))
