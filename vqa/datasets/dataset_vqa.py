import collections
import h5py
import json
import os
import numpy as np

from PIL import Image

from util import log

RANDOM_STATE = np.random.RandomState(123)
IMAGE_WIDTH = 540
IMAGE_HEIGHT = 540

MAX_ROI_NUM = 50


class Dataset(object):

    def __init__(self, ids, dataset_path, image_dir, vfeat_path, vocab_path,
                 is_train=True, name='default'):
        self.name = name

        self._ids = list(ids)
        self.image_dir = image_dir
        self.is_train = is_train

        self.width = IMAGE_WIDTH
        self.height = IMAGE_HEIGHT

        self.vocab = json.load(open(vocab_path, 'r'))

        file_name = os.path.join(dataset_path, 'data.hdf5')
        log.info('Reading {} ...'.format(file_name))

        self.data = h5py.File(file_name, 'r')
        self.data_info = self.data['data_info']

        self.vfeat = h5py.File(vfeat_path, 'r')

        log.info('Reading Done {}'.format(file_name))

    def get_config(self):
        config = collections.namedtuple('dataset_config', [])
        config.image_width = IMAGE_WIDTH
        config.image_height = IMAGE_HEIGHT
        config.max_roi_num = MAX_ROI_NUM
        return config

    def get_data(self, id):
        entry = self.data[id]

        image_path = entry['image_path'].value
        image_id = image_path.replace('/', '-')

        vfeat_entry = self.vfeat[image_id]

        # Image
        o_image = Image.open(os.path.join(self.image_dir, image_path))
        o_w, o_h = o_image.size
        image = np.array(
            o_image.resize([self.width, self.height]).convert('RGB'),
            dtype=np.float32)

        box = vfeat_entry['box'].value
        num_box = np.array(vfeat_entry['num_box'].value, dtype=np.int32)
        V_ft = vfeat_entry['vfeat'].value

        q_intseq = entry['question_intseq'].value
        q_intseq_len = np.array(len(q_intseq), dtype=np.int32)

        answer_id = np.array(entry['answer_id'].value, dtype=np.int32)

        """
        Returns:
            - image: resized rgb image (scale: [0, 255])
            - box: all set of boxes used for roi pooling (x1y1x2y2 format)
            - num_box: number of boxes
            - V_ft: pre-extracted visual features [num_box x V_DIM]
            - q_intseq: intseq tokens of question
            - q_intseq_len: length of intseq question
            - answer_id: ground truth index in answer candidates
        """
        returns = {
            'image': image,
            'box': box,
            'num_box': num_box,
            'V_ft': V_ft,
            'q_intseq': q_intseq,
            'q_intseq_len': q_intseq_len,
            'answer_id': answer_id
        }
        return returns

    def get_data_shapes(self):
        data_shapes = {
            'image': [self.height, self.width, 3],
            'box': [None, 4],
            'num_box': (),
            'V_ft': [None, 512],
            'q_intseq': [None],
            'q_intseq_len': (),
            'answer_id': (),
        }
        return data_shapes

    def get_data_types(self):
        data_types = {
            'image': np.float32,
            'box': np.float32,
            'num_box': np.int32,
            'V_ft': np.float32,
            'q_intseq': np.int32,
            'q_intseq_len': np.int32,
            'answer_id': np.int32,
        }
        return data_types

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset ({}, {} examples)'.format(self.name, len(self))


def create_default_splits(dataset_path, image_dir, vfeat_path, vocab_path,
                          is_train=True):
    ids_train, ids_val, ids_testval, ids_test = all_ids(dataset_path,
                                                        is_train=is_train)
    dataset_train = Dataset(ids_train, dataset_path, image_dir, vfeat_path,
                            vocab_path, is_train=is_train, name='train')
    dataset_val = Dataset(ids_val, dataset_path, image_dir, vfeat_path,
                          vocab_path, is_train=is_train, name='val')
    dataset_testval = Dataset(ids_testval, dataset_path, image_dir, vfeat_path,
                              vocab_path, is_train=is_train, name='test')
    dataset_test = Dataset(ids_test, dataset_path, image_dir, vfeat_path,
                           vocab_path, is_train=is_train, name='test')
    return {
        'train': dataset_train,
        'val': dataset_val,
        'testval': dataset_testval,
        'test': dataset_test,
    }


def all_ids(dataset_path, is_train=True):
    with h5py.File(os.path.join(dataset_path, 'data.hdf5'), 'r') as f:
        num_train = int(f['data_info']['num_train'].value)
        num_val = int(f['data_info']['num_val'].value)
        num_testval = int(f['data_info']['num_test-val'].value)
        num_test = int(f['data_info']['num_test'].value)

    with open(os.path.join(dataset_path, 'id.txt'), 'r') as fp:
        ids_total = fp.read().splitlines()

    start_train = 0
    start_val = start_train + num_train
    start_testval = start_val + num_val
    start_test = start_testval + num_test

    ids_train = ids_total[start_train: start_train + num_train]
    ids_val = ids_total[start_val: start_val + num_val]
    ids_testval = ids_total[start_testval: start_testval + num_testval]
    ids_test = ids_total[start_test: start_test + num_test]

    if is_train:
        RANDOM_STATE.shuffle(ids_train)
        RANDOM_STATE.shuffle(ids_val)
        RANDOM_STATE.shuffle(ids_testval)
        RANDOM_STATE.shuffle(ids_test)

    return ids_train, ids_val, ids_testval, ids_test
