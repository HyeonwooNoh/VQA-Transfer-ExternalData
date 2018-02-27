import h5py
import os
import numpy as np

from PIL import Image

from util import log

RANDOM_STATE = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, dataset_path, image_dir, num_k,
                 width=224, height=224, name='default', is_train=True):
        self._ids = list(ids)
        self.image_dir = image_dir
        self.num_k = num_k  # number of positive + negative objects
        self.width = width
        self.height = height
        self.name = name
        self.is_train = is_train

        file_name = os.path.join(dataset_path, 'data.hdf5')
        log.info('Reading {} ...'.format(file_name))

        self.data = h5py.File(file_name, 'r')
        self.data_info = self.data['data_info']
        self.objects_intseq = self.data_info['objects_intseq'].value
        self.objects_intseq_len = self.data_info['objects_intseq_len'].value
        self.num_objects = int(self.data_info['num_unique_objects'].value)
        self.objects_ids = list(range(self.num_objects))
        self.max_name_len = int(self.data_info['max_name_length'].value)

        log.info('Reading Done {}'.format(file_name))

    def get_data(self, id):
        """
        Returns:
            image: [height, width, channel]
            sampled_objects: [num_k, max_name_len]  (first is positive object)
            sampled_objects_len: [num_k]
            ground_truth: [num_k]  (one-hot vector with 1 as the first entry)

        """
        image_id, id = id.split()
        entry = self.data[image_id][id]

        name_ids = list(entry['name_ids'].value)
        negative_ids = list(set(self.objects_ids) - set(name_ids))
        RANDOM_STATE.shuffle(negative_ids)

        sampled_ids = (name_ids + negative_ids)[:self.num_k]
        sampled_objects = np.take(self.objects_intseq, sampled_ids, axis=0)
        sampled_objects_len = np.take(
            self.objects_intseq_len, sampled_ids, axis=0)
        num_pos, num_neg = len(name_ids), self.num_k - len(name_ids)
        ground_truth = np.array([1] * num_pos + [0] * num_neg, dtype=np.float32)

        x, y = entry['x'].value, entry['y'].value
        w, h = entry['w'].value, entry['h'].value

        image_path = os.path.join(self.image_dir, '{}.jpg'.format(image_id))
        image = np.array(
            Image.open(image_path).crop([x, y, x + w, y + h])
            .resize([self.width, self.height]).convert('RGB'), dtype=np.float32
        ) / 128.0 - 1.0  # normalize to [-1, 1]

        return image, sampled_objects, sampled_objects_len, ground_truth

    def get_data_shapes(self):
        data_shapes = {
            'image': [self.height, self.width, 3],
            'sampled_objects': [self.num_k, self.max_name_len],
            'sampled_objects_len': [self.num_k],
            'ground_truth': [self.num_k],
        }
        return data_shapes

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset ({}, {} examples)'.format(self.name, len(self))


def create_default_splits(dataset_path, image_dir, num_k, is_train=True):
    """
    Args:
        num_k: number positive + negative object names sampled for training
    """
    ids_train, ids_test, ids_val = all_ids(dataset_path, is_train=is_train)

    dataset_train = Dataset(ids_train, dataset_path, image_dir, num_k,
                            width=224, height=224,
                            name='train', is_train=is_train)
    dataset_test = Dataset(ids_test, dataset_path, image_dir, num_k,
                           width=224, height=224,
                           name='test', is_train=is_train)
    dataset_val = Dataset(ids_val, dataset_path, image_dir, num_k,
                          width=224, height=224,
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
