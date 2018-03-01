import h5py
import json
import os
import numpy as np

from PIL import Image

from util import log

RANDOM_STATE = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, dataset_path, image_dir, vocab_path,
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
        self.num_unique_descriptions = \
            int(self.data_info['num_unique_descriptions'].value)
        self.max_len = int(self.data_info['max_length'].value)
        log.info('Reading Done {}'.format(file_name))

    def get_data(self, id):
        """
        Returns:
            image: [height, width, channel]
            region_description: [max_len + 1]  (including <e>)
            region_description_len: () (length not including <e>)
        """
        image_id, id = id.split()
        entry = self.data[image_id][id]

        desc = entry['description'].vaue
        padded_desc = np.zeros([self.max_len + 1], dtype=np.int32)
        desc_len = len(desc)
        padded_desc[:desc_len] = desc
        padded_desc[desc_len] = self.vocab['dict']['<e>']  # end token

        x, y = entry['x'].value, entry['y'].value
        w, h = entry['w'].value, entry['h'].value

        image_path = os.path.join(self.image_dir, '{}.jpg'.format(image_id))
        image = np.array(
            Image.open(image_path).crop([x, y, x + w, y + h])
            .resize([self.width, self.height]).convert('RGB'), dtype=np.float32
        )

        return image, padded_desc, desc_len

    def get_data_shapes(self):
        data_shapes = {
            'image': [self.height, self.width, 3],
            'region_description': [self.max_len + 1],
            'region_description_len': [],
        }
        return data_shapes

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset ({}, {} examples)'.format(self.name, len(self))


def create_default_splits(dataset_path, image_dir, vocab_path, is_train=True):
    ids_train, ids_test, ids_val = all_ids(dataset_path, is_train=is_train)

    dataset_train = Dataset(ids_train, dataset_path, image_dir, vocab_path,
                            width=224, height=224,
                            name='train', is_train=is_train)
    dataset_test = Dataset(ids_test, dataset_path, image_dir, vocab_path,
                           width=224, height=224,
                           name='test', is_train=is_train)
    dataset_val = Dataset(ids_val, dataset_path, image_dir, vocab_path,
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
