import collections
import h5py
import json
import os
import numpy as np

from PIL import Image

from util import log, box_utils

RANDOM_STATE = np.random.RandomState(123)
IMAGE_WIDTH = 540
IMAGE_HEIGHT = 540

MAX_ROI_NUM = 50

DENSECAP_FILENAME = 'results_original_size.hdf5'


class Dataset(object):

    def __init__(self, image_paths, image_dir, densecap_dir,
                 is_train=True, name='default'):
        self.name = name

        self._ids = list(range(len(image_paths)))
        self.image_paths = image_paths

        self.image_dir = image_dir
        self.densecap_dir = densecap_dir
        self.is_train = is_train

        self.width = IMAGE_WIDTH
        self.height = IMAGE_HEIGHT

        self.max_roi_num = MAX_ROI_NUM

        self.densecap = {}
        for split in ['train2014', 'val2014', 'test2015']:
            densecap_path = os.path.join(
                self.densecap_dir, split, DENSECAP_FILENAME)
            self.densecap[split] = h5py.File(densecap_path, 'r')

    def get_config(self):
        config = collections.namedtuple('dataset_config', [])
        config.image_width = IMAGE_WIDTH
        config.image_height = IMAGE_HEIGHT
        config.max_roi_num = MAX_ROI_NUM
        return config

    def get_data(self, id):
        image_path = self.image_paths[id]

        # Image
        o_image = Image.open(os.path.join(self.image_dir, image_path))
        o_w, o_h = o_image.size
        image = np.array(
            o_image.resize([self.width, self.height]).convert('RGB'),
            dtype=np.float32)
        frac_x = self.width / float(o_w)
        frac_y = self.height / float(o_h)

        # Box
        def preprocess_box(box):
            return box_utils.xywh_to_x1y1x2y2(box_utils.scale_boxes_xywh(
                box.astype(np.float32), [frac_x, frac_y]))

        split = image_path.split('/')[0]
        image_id = image_path.replace('/', '-')
        box = preprocess_box(
            self.densecap[split][image_id].value)
        box = box[:MAX_ROI_NUM]
        normal_box = box_utils.normalize_boxes_x1y1x2y2(
            box, self.width, self.height)
        num_box = np.array(box.shape[0], dtype=np.int32)

        image_id = np.array(list(image_id), dtype=np.str)
        image_id_len = len(image_id)

        """
        Returns:
            * image and box:
                - image: resized rgb image (scale: [0, 255])
                - box: all set of boxes used for roi pooling (x1y1x2y2 format)
                - normal_box: normalized (y1x1y2x2 [0, 1]) box
                - num_box: number of boxes
                - image_id: list form of
                  ex) train2014-COCO_train2014_00000045123.jpg
                - image_id_len: length of image_id
        """
        returns = {
            'image': image,
            'box': box,
            'normal_box': normal_box,
            'num_box': num_box,
            'image_id': image_id,
            'image_id_len': image_id_len,
        }
        return returns

    def get_data_shapes(self):
        data_shapes = {
            'image': [self.height, self.width, 3],
            'box': [None, 4],
            'normal_box': [None, 4],
            'num_box': (),
            'image_id': [None],
            'image_id_len': (),
        }
        return data_shapes

    def get_data_types(self):
        data_types = {
            'image': np.float32,
            'box': np.float32,
            'normal_box': np.float32,
            'num_box': np.int32,
            'image_id': np.str,
            'image_id_len': np.int32,
        }
        return data_types


    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset ({}, {} examples)'.format(self.name, len(self))


def create_dataset(used_image_path, image_dir, densecap_dir, is_train=True):
    image_paths = all_ids(used_image_path, is_train=is_train)

    dataset = Dataset(image_paths, image_dir, densecap_dir,
                      is_train=is_train, name='dataset')
    return dataset


def all_ids(used_image_path, is_train=True):
    image_paths = open(used_image_path, 'r').read().splitlines()
    return image_paths
