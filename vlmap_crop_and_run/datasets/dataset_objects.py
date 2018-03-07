import h5py
import json
import os
import numpy as np

from PIL import Image

from util import log

RANDOM_STATE = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, dataset_path, image_dir, vocab_path, num_k,
                 width=224, height=224, name='default', is_train=True):
        self._ids = list(ids)
        self.image_dir = image_dir
        self.width = width
        self.height = height
        self.name = name
        self.is_train = is_train

        self.vocab = json.load(open(vocab_path, 'r'))

        file_name = os.path.join(dataset_path, 'data.hdf5')
        log.info('Reading {} ...'.format(file_name))

        self.data = h5py.File(file_name, 'r')
        self.data_info = self.data['data_info']
        self.objects_intseq = self.data_info['objects_intseq'].value
        self.objects_intseq_len = self.data_info['objects_intseq_len'].value
        self.objects_name = []
        for obj, obj_len in zip(self.objects_intseq, self.objects_intseq_len):
            self.objects_name.append(
                ' '.join([self.vocab['vocab'][i] for i in obj[:obj_len]]))
        self.objects_name = np.array(self.objects_name)
        self.num_objects = int(self.data_info['num_unique_objects'].value)
        self.objects_ids = list(range(self.num_objects))
        self.max_name_len = int(self.data_info['max_name_length'].value)

        if is_train:
            self.num_k = num_k  # number of positive + negative objects
        else: self.num_k = self.num_objects

        log.info('Reading Done {}'.format(file_name))

    def get_data(self, id):
        """
        Returns:
            image: [height, width, channel]
            sampled_objects: [num_k, max_name_len]  (first is positive object)
            sampled_objects_len: [num_k]
            ground_truth: [num_k]  (one-hot vector with 1 as the first entry
            sampled_objects_name: [num_k]
        """
        image_id, id = id.split()
        entry = self.data[image_id][id]

        if self.is_train:
            name_ids = list(entry['name_ids'].value)
            negative_ids = list(set(self.objects_ids) - set(name_ids))
            RANDOM_STATE.shuffle(negative_ids)

            sampled_ids = (name_ids + negative_ids)[:self.num_k]
            sampled_objects = np.take(self.objects_intseq, sampled_ids, axis=0)
            sampled_objects_len = np.take(
                self.objects_intseq_len, sampled_ids, axis=0)
            sampled_objects_name = []
            for obj, obj_len in zip(sampled_objects, sampled_objects_len):
                sampled_objects_name.append(
                    ' '.join([self.vocab['vocab'][i] for i in obj[:obj_len]]))
            sampled_objects_name = np.array(sampled_objects_name)
            num_pos, num_neg = len(name_ids), self.num_k - len(name_ids)
            ground_truth = np.array([1] * num_pos + [0] * num_neg, dtype=np.float32)
        else:
            name_ids = list(entry['name_ids'].value)
            sampled_objects = self.objects_intseq
            sampled_objects_len = self.objects_intseq_len
            sampled_objects_name = self.objects_name
            ground_truth = np.zeros([self.num_objects], dtype=np.float32)
            for name_id in name_ids:
                ground_truth[name_id] = 1.0

        x, y = entry['x'].value, entry['y'].value
        w, h = entry['w'].value, entry['h'].value

        image_path = os.path.join(self.image_dir, '{}.jpg'.format(image_id))
        image = np.array(
            Image.open(image_path).crop([x, y, x + w, y + h])
            .resize([self.width, self.height]).convert('RGB'), dtype=np.float32
        )

        return image, sampled_objects, sampled_objects_len, ground_truth,\
            sampled_objects_name

    def get_data_shapes(self):
        data_shapes = {
            'image': [self.height, self.width, 3],
            'sampled_objects': [self.num_k, self.max_name_len],
            'sampled_objects_len': [self.num_k],
            'ground_truth': [self.num_k],
            'sampled_objects_name': [self.num_k],
        }
        return data_shapes

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset ({}, {} examples)'.format(self.name, len(self))


def create_default_splits(dataset_path, image_dir, vocab_path,
                          num_k, is_train=True):
    """
    Args:
        num_k: number positive + negative object names sampled for training
    """
    ids_train, ids_test, ids_val = all_ids(dataset_path, is_train=is_train)

    dataset_train = Dataset(ids_train, dataset_path, image_dir, vocab_path,
                            num_k, width=224, height=224,
                            name='train', is_train=is_train)
    dataset_test = Dataset(ids_test, dataset_path, image_dir, vocab_path,
                           num_k, width=224, height=224,
                           name='test', is_train=is_train)
    dataset_val = Dataset(ids_val, dataset_path, image_dir, vocab_path,
                          num_k, width=224, height=224,
                          name='val', is_train=is_train)
    return {
        'train': dataset_train,
        'test': dataset_test,
        'val': dataset_val
    }


def all_ids(dataset_path, is_train=True):
    with h5py.File(os.path.join(dataset_path, 'data.hdf5'), 'r') as f:
        num_train = int(f['data_info']['num_train'].value)
        num_test = int(f['data_info']['num_test'].value)
        num_val = int(f['data_info']['num_val'].value)

    with open(os.path.join(dataset_path, 'id.txt'), 'r') as fp:
        ids_total = fp.read().splitlines()

    ids_train = ids_total[:num_train]
    ids_test = ids_total[num_train: num_train + num_test]
    ids_val = ids_total[num_train + num_test: num_train + num_test + num_val]

    if is_train:
        RANDOM_STATE.shuffle(ids_train)
        RANDOM_STATE.shuffle(ids_test)
        RANDOM_STATE.shuffle(ids_val)

    return ids_train, ids_test, ids_val
