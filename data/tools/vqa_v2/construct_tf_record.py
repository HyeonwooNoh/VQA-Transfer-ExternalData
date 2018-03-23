import argparse
import h5py
import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from PIL import Image

from util import log, tf_util

RANDOM_STATE = np.random.RandomState(123)
IMAGE_WIDTH = 540
IMAGE_HEIGHT = 540

MAX_ROI_NUM = 50

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_dir', type=str, default='data/VQA_v2/images',
                    help=' ')
parser.add_argument('--qa_split_dir', type=str,
                    default='data/preprocessed/vqa_v2'
                    '/qa_split_thres1_500_thres2_50', help=' ')
parser.add_argument('--dataset_name', type=str, default='data', help=' ')
parser.add_argument('--vfeat_name', type=str, default='used_vfeat.hdf5',
                    help=' ')
parser.add_argument('--vocab_name', type=str, default='vocab.json', help=' ')
parser.add_argument('--tf_record_dir', type=str, default='tf_record', help=' ')
parser.add_argument('--split', type=str, default='train',
                    choices=['train', 'val', 'testval', 'test'], help=' ')
parser.add_argument('--num_record_per_shard', type=int, default=1024, help=' ')
config = parser.parse_args()

config.dataset_dir = os.path.join(config.qa_split_dir, config.dataset_name)
config.vfeat_path = os.path.join(config.qa_split_dir, config.vfeat_name)
config.vocab_path = os.path.join(config.qa_split_dir, config.vocab_name)

config.tf_record_dir = os.path.join(
    config.qa_split_dir, config.tf_record_dir, config.split)
if not os.path.exists(config.tf_record_dir):
    log.warn('create directory: {}'.format(config.tf_record_dir))
    os.makedirs(config.tf_record_dir)
else:
    raise ValueError('The directory {} already exists. Do not overwrite.'.format(
        config.tf_record_dir))

data = h5py.File(os.path.join(config.dataset_dir, 'data.hdf5'), 'r')
vfeat = h5py.File(config.vfeat_path, 'r')

data_info = data['data_info']
num_data = {
    'train': int(data_info['num_train'].value),
    'val': int(data_info['num_val'].value),
    'testval': int(data_info['num_test-val'].value),
    'test': int(data_info['num_test'].value),
}
ids_total = open(os.path.join(config.dataset_dir, 'id.txt'),
                 'r').read().splitlines()
start_train = 0
start_val = start_train + num_data['train']
start_testval = start_val + num_data['val']
start_test = start_testval + num_data['testval']

ids = {
    'train': ids_total[start_train: start_train + num_data['train']],
    'val': ids_total[start_val: start_val + num_data['val']],
    'testval': ids_total[start_testval: start_testval + num_data['testval']],
    'test': ids_total[start_test: start_test + num_data['test']],
}

RANDOM_STATE.shuffle(ids['train'])
RANDOM_STATE.shuffle(ids['val'])
RANDOM_STATE.shuffle(ids['testval'])
RANDOM_STATE.shuffle(ids['test'])

session_config = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 0})
sess = tf.Session(config=session_config)

with tf.device('/cpu:0'):
    image_placeholder = tf.placeholder(tf.uint8)
    png_encoded = tf.image.encode_png(image_placeholder)

log.warn('write tf_record of {} data: {}'.format(
    config.split, config.tf_record_dir))
max_q_len = data['data_info']['max_q_len'].value
max_box_num = vfeat['data_info']['max_box_num'].value
num_shards = len(ids[config.split]) / config.num_record_per_shard + 1
for i, id in enumerate(tqdm(ids[config.split], desc='{} ids'.format(config.split))):
    if i % config.num_record_per_shard == 0:
        shard_id = int(i / config.num_record_per_shard)
        shard_name = '{}-{:05d}-of-{:05d}'.format(
            config.split, shard_id, num_shards)
        shard_path = os.path.join(config.tf_record_dir, shard_name)
        tf_record_writer = tf.python_io.TFRecordWriter(shard_path)

    entry = data[id]

    image_path = entry['image_path'].value
    image_id = image_path.replace('/', '-')

    vfeat_entry = vfeat[image_id]

    # Image
    o_image = Image.open(os.path.join(config.image_dir, image_path))
    o_w, o_h = o_image.size
    image = np.array(
        o_image.resize([IMAGE_WIDTH, IMAGE_HEIGHT]).convert('RGB'),
        dtype=np.uint8)
    image_str = sess.run(png_encoded, {image_placeholder: image})

    num_box = np.array(vfeat_entry['num_box'].value, dtype=np.int32)

    box = vfeat_entry['box'].value
    pad_box = np.zeros([max_box_num - num_box, box.shape[1]], dtype=box.dtype)
    box = np.concatenate([box, pad_box], axis=0)

    V_ft = vfeat_entry['vfeat'].value
    pad_V_ft = np.zeros([max_box_num - num_box, V_ft.shape[1]], dtype=V_ft.dtype)
    V_ft = np.concatenate([V_ft, pad_V_ft], axis=0)

    q_intseq = entry['question_intseq'].value
    q_intseq_len = np.array(len(q_intseq), dtype=np.int32)
    q_intseq_pad = np.zeros([max_q_len - q_intseq_len], dtype=q_intseq.dtype)
    q_intseq = np.concatenate([q_intseq, q_intseq_pad], axis=0)

    answer_id = np.array(entry['answer_id'].value, dtype=np.int32)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'qid': tf_util.int64_feature(int(id)),
        'image_id': tf_util.bytes_feature(str(image_id)),
        'image/encoded': tf_util.bytes_feature(image_str),
        'image/format': tf_util.bytes_feature('png'),
        'image/height': tf_util.int64_feature(IMAGE_HEIGHT),
        'image/width': tf_util.int64_feature(IMAGE_WIDTH),
        'box/list': tf_util.float_feature(list(box.reshape([-1]))),
        'box/shape': tf_util.int64_feature(list(box.shape)),
        'num_box': tf_util.int64_feature(num_box),
        'V_ft/list': tf_util.float_feature(list(V_ft.reshape([-1]))),
        'V_ft/shape': tf_util.int64_feature(list(V_ft.shape)),
        'q_intseq/list': tf_util.int64_feature(list(q_intseq)),
        'q_intseq/len': tf_util.int64_feature(q_intseq_len),
        'answer_id': tf_util.int64_feature(answer_id),
    }))
    tf_record_writer.write(tf_example.SerializeToString())

data.close()
vfeat.close()
