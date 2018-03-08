import h5py
import json
import os
import numpy as np

from PIL import Image

from util import log, box_utils

RANDOM_STATE = np.random.RandomState(123)
IMAGE_WIDTH = 540
IMAGE_HEIGHT = 540

NUM_K = 500  # number of entry used for classification.

MAX_USED_BOX = 100
MAX_REGION_BOX = 30
MAX_OBJECT_BOX = 10

class Dataset(object):

    def __init__(self, ids, dataset_path, image_dir, vocab_path, is_train=True,
                 name='default'):
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

        # entry is names for all classes (used for classification objective)
        self.entry = {
            'obj': self.data_info['objects_intseq'].value,
            'attr': self.data_info['attributes_intseq'].value,
            'rel': self.data_info['relationships_intseq'].value}
        self.entry_len = {
            'obj': self.data_info['objects_intseq_len'].value,
            'attr': self.data_info['attributes_intseq_len'].value,
            'rel': self.data_info['relationships_intseq_len'].value}
        self.entry_name = {}
        for key in ['obj', 'attr', 'rel']:
            en = []
            for e, el in zip(self.entry[key], self.entry_len[key]):
                en.append(' '.join([self.vocab['vocab'][i] for i in e[:el]]))
            self.entry_name[key] = np.array(en)

        self.num_entry = {
            'obj': int(self.data_info['num_unique_objects'].value),
            'attr': int(self.data_info['num_unique_attributes'].value),
            'rel': int(self.data_info['num_unique_relationships'].value),
            'region': int(self.data_info['num_unique_descriptions'].value)}
        self.max_len = {
            'obj': int(self.data_info['max_object_name_len'].value),
            'attr': int(self.data_info['max_attribute_name_len'].value),
            'rel': int(self.data_info['max_relationship_name_len'].value),
            'region': int(self.data_info['max_description_length'].value)}
        self.max_num_per_box = {
            'obj': int(self.data_info['max_object_num_per_box'].value),
            'attr': int(self.data_info['max_attribute_num_per_box'].value),
            'rel': int(self.data_info['max_relationship_num_per_box'].value)}

        log.info('Reading Done {}'.format(file_name))

    def get_data(self, id):
        """
        Returns:
            image: [height, width, channel]

            * following boxes will be concatenated
            pos_box: [N, 4 (x1, y1, x2, y2)]
                - This is set of densecap boxes used by at least one task
            neg_box: [M, 4 (x1, y1, x2, y2)]
                - This is for metric learning - "finding box from description"
                - N + M can be a constant for effective batching.
            region_gt_box
            obj_gt_box
            attr_gt_box
            rel_gt_box

            * following data will be used for description / blank-fill
            region_box_idx [None]
            region_desc [None, Len] (including <e>)
            region_desc_len [None] (length including <e>)
            region_blank_desc [None, Len] (including <e>)
            region_blank_desc_len [None] (length including <e>)

            (region_box_idx will be used as ground truth for image retrieval task)
            region_box_candidates_idx [None, num_k]
            region_box_gt [None, num_k] (one-hot)

            * following data will be used for text retrieval
            region_box_idx [None]
            region_desc [None, Len] (including <e>)
            region_desc_len [None] (length including <e>)
            (we don't have to explicitly make and return this data. This could
            be handled by the model)

            * following data will be used for classification
            obj_box_idx [None]
            obj_name_list: [None, num_k]
            obj_name_len: [None, num_k]
            obj_gt: [None, num_k] (one-hot)

            attr_box_idx [None]
            attr_name_list: [None, num_k]
            attr_name_len: [None, num_k]
            attr_gt: [None, num_k] (one-hot)

            rel_box_idx [None]
            rel_name_list: [None, num_k]
            rel_name_len: [None, num_k]
            rel_gt: [None, num_k] (one-hot)

            positive_box
            sampled_objects: [num_k, max_name_len]  (first is positive object)
            sampled_objects_len: [num_k]
            ground_truth: [num_k]  (one-hot vector with 1 as the first entry
            sampled_objects_name: [num_k]
        """
        entry = self.data[id]

        # Image
        image_path = os.path.join(self.image_dir, '{}.jpg'.format(image_id))
        o_image = Image.open(image_path)
        o_w, o_h = o_image.size
        image = np.array(
            o_image.resize([self.width, self.height]).convert('RGB'),
            dtype=np.float32)
        frac_x = self.width / float(o_w)
        frac_y = self.height / float(o_h)

        def preprocess_box(box):
            return box_utils.xywh_to_x1y1x2y2(box_utils.scale_boxes_xywh(
                box.astype(np.float32), [frac_x, frac_y]))

        # Current dataset implementation uses image coordinate.
        box = preprocess_box(entry['box_xywh'].value)

        # [1] Add positive densecap boxes
        pos_box_idx = entry['positive_box_idx'].value
        num_pos_box = len(pos_box_idx)
        if num_pos_box > 0: used_box = np.take(box, pos_box_idx, axis=0)
        else: used_box = np.zeros([0, 4], dtype=np.float32)

        box_idx_start_region = used_box.shape[0]

        # [2] Add no-assigned regions
        asn_region_idx = entry['asn_region_idx'].value
        no_asn_region_idx = entry['no_asn_region_idx'].value

        num_asn_region = len(asn_region_idx)
        num_no_asn_region = len(no_asn_region_idx)
        # how to determine the number of no_asn regions
        # 1. left 1 object, 1 attribute, 1 relations
        # 2. fill max_region_box limitation
        # 3. use all labeled regions
        used_num_no_asn_region = min(MAX_USED_BOX - box_idx_start_region - 3,
                                     MAX_REGION_BOX - num_asn_region,
                                     num_no_asn_region)
        used_no_asn_region_selector = list(range(num_no_asn_region))
        RANDOM_STATE.shuffle(used_no_asn_region_selector)
        used_no_asn_region_selector = \
            used_no_asn_region_selector[:num_no_asn_region]

        if len(used_no_asn_region_selector) > 0:
            used_no_asn_region_idx = np.take(
                no_asn_region_idx, used_no_asn_region_selector, axis=0)
        else: used_no_asn_region_idx = np.array([], dtype=np.int32)

        region_box = preprocess_box(entry['region_xywh'].value)
        if len(used_no_asn_region_idx) > 0:
            no_asn_region_box = np.take(region_box, used_no_asn_region_idx, axis=0)
        else: no_asn_region_box = np.zeros([0, 4], dtype=np.float32)

        used_box = np.concatenate([used_box, no_asn_region_box], axis=0)

        box_idx_start_object = used_box.shape[0]

        # [3] Add no-assigned object
        asn_object_idx = entry['asn_object_idx'].value
        no_asn_object_idx = entry['no_asn_object_idx'].value

        num_asn_object = len(asn_object_idx)
        num_no_asn_object = len(no_asn_object_idx)

        used_num_no_asn_object = min(MAX_USED_BOX - box_idx_start_object - 2,
                                     MAX_OBJECT_BOX - num_asn_object,
                                     num_no_asn_object)
        used_no_asn_object_selector = list(range(num_no_asn_object))
        RANDOM_STATE.shuffle(used_no_asn_object_selector)
        used_no_asn_object_selector = \
            used_no_asn_object_selector[:num_no_asn_object]

        if len(used_no_asn_object_selector) > 0:
            used_no_asn_object_idx = np.take(
                no_asn_object_idx, used_no_asn_object_selector, axis=0)
        else: used_no_asn_object_idx = np.array([], dtype=np.int32)

        object_box = preprocess_box(entry['object_xywh'].value)
        if len(used_no_asn_object_idx) > 0:
            no_asn_object_box = np.take(object_box, used_no_asn_object_idx, axis=0)
        else: no_asn_object_box = np.zeros([0, 4], dtype=np.float32)

        used_box = np.concatenate([used_box, no_asn_object_box], axis=0)

        box_idx_start_attribute = used_box.shape[0]

        # [4] Add no-assigned attribute




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


def create_default_splits(dataset_path, image_dir, vocab_path, is_train=True):
    ids_train, ids_test, ids_val = all_ids(dataset_path, is_train=is_train)

    dataset_train = Dataset(ids_train, dataset_path, image_dir, vocab_path,
                            is_train=is_train, name='train')
    dataset_test = Dataset(ids_test, dataset_path, image_dir, vocab_path,
                           is_train=is_train, name='test')
    dataset_val = Dataset(ids_val, dataset_path, image_dir, vocab_path,
                          is_train=is_train, name='val')
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
