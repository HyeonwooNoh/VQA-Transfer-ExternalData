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
IMAGE_RETRIEVAL_K = 10
LANGUAGE_RETRIEVAL_K = 10

MAX_USED_BOX = 60
MAX_BOX_PER_ENTRY = {
    'region': 30,
    'object': 10,
    'attribute': 10,
    'relationship': 10
}

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
            'object': self.data_info['objects_intseq'].value,
            'attribute': self.data_info['attributes_intseq'].value,
            'relationship': self.data_info['relationships_intseq'].value}
        self.entry_len = {
            'object': self.data_info['objects_intseq_len'].value,
            'attribute': self.data_info['attributes_intseq_len'].value,
            'relationship': self.data_info['relationships_intseq_len'].value}
        self.entry_name = {}
        for key in ['object', 'attribute', 'relationship']:
            en = []
            for e, el in zip(self.entry[key], self.entry_len[key]):
                en.append(' '.join([self.vocab['vocab'][i] for i in e[:el]]))
            self.entry_name[key] = np.array(en)

        self.num_entry = {
            'object': int(self.data_info['num_unique_objects'].value),
            'attribute': int(self.data_info['num_unique_attributes'].value),
            'relationship': int(self.data_info['num_unique_relationships'].value),
            'region': int(self.data_info['num_unique_descriptions'].value)}
        self.max_len = {
            'object': int(self.data_info['max_object_name_len'].value),
            'attribute': int(self.data_info['max_attribute_name_len'].value),
            'relationship': int(self.data_info['max_relationship_name_len'].value),
            'region': int(self.data_info['max_description_length'].value)}
        self.max_num_per_box = {
            'object': int(self.data_info['max_object_num_per_box'].value),
            'attribute': int(self.data_info['max_attribute_num_per_box'].value),
            'relationship': int(self.data_info['max_relationship_num_per_box'].value)}

        log.info('Reading Done {}'.format(file_name))

    def get_data(self, id):
        """
        Returns:
            * image and box:
                - image: resized rgb image (scale: [0, 255])
                - box: all set of boxes used for roi pooling (x1y1x2y2 format)
            * description:
                - desc: region description ground truths
                - desc_len: region description lengths
                - desc_box_idx: index of description corresponded boxes
                - num_used_desc: number of descriptions except for padding
            * blank fill:
                - blank_desc: region description with random blank
                - blank_desc_len: length of blank_desc
            * language retrieval (lr) and image retrieval (ir):
                - lr_desc_idx: description candidates index for matching
                - lr_gt: ground truth retrieval results (one-hot)
                - ir_box_idx: candidate boxes index
                - ir_gt: ground truth retrieval results (one-hot)
            * entry [object, attribute, relationship] classification:
                - {}_box_idx: box index for entry classification
                - {}_num_used_box: number of used boxes (for masking loss)
                - {}_candidate: intseq of candidate entry names
                - {}_candidate_len: intseq length  of candidate entry names
                - {}_candidate_name: string of candidate entry (for debugging)
                - {}_selection_gt: gt for entry selection task
        """
        entry = self.data[id]

        # Image
        image_path = os.path.join(self.image_dir, '{}.jpg'.format(id))
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

        box_idx_start = {}
        asn_idx = {}
        used_no_asn_idx = {}
        for i, key in enumerate(['region', 'object', 'attribute', 'relationship']):
            box_idx_start[key] = used_box.shape[0]

            # [2, 3, 4] Add no-assigned regions to used box list
            asn_idx[key] = entry['asn_{}_idx'.format(key)].value
            no_asn_idx = entry['no_asn_{}_idx'.format(key)].value

            num_asn = len(asn_idx[key])
            num_no_asn = len(no_asn_idx)

            # how to determine the number of no_asn regions
            # 1. left 1 object, 1 attribute, 1 relations
            # 2. fill max_region_box limitation
            # 3. use all labeled regions
            num_used_no_asn = min(MAX_USED_BOX - box_idx_start[key] - (3 - i),
                                  MAX_BOX_PER_ENTRY[key] - num_asn,
                                  num_no_asn)
            used_no_asn_selector = list(range(num_no_asn))
            RANDOM_STATE.shuffle(used_no_asn_selector)
            used_no_asn_selector = used_no_asn_selector[:num_used_no_asn]

            if len(used_no_asn_selector) > 0:
                used_no_asn_idx[key] = np.take(
                    no_asn_idx, used_no_asn_selector, axis=0)
            else: used_no_asn_idx[key] = np.array([], dtype=np.int32)

            entry_box = preprocess_box(entry['{}_xywh'.format(key)].value)
            if len(used_no_asn_idx[key]) > 0:
                no_asn_box = np.take(entry_box, used_no_asn_idx[key], axis=0)
            else: no_asn_box = np.zeros([0, 4], dtype=np.float32)
            used_box = np.concatenate([used_box, no_asn_box], axis=0)

        # [5] Add negative boxes
        box_idx_start['neg_box'] = used_box.shape[0]
        neg_box_idx = entry['negative_box_idx'].value
        num_neg_box = len(neg_box_idx)
        num_used_neg_box = min(MAX_USED_BOX - box_idx_start['neg_box'],
                               num_neg_box)
        used_neg_box_selector = list(range(num_neg_box))
        RANDOM_STATE.shuffle(used_neg_box_selector)
        used_neg_box_selector = used_neg_box_selector[:num_used_neg_box]

        if len(used_neg_box_selector) > 0:
            used_neg_box_idx = np.take(
                neg_box_idx, used_neg_box_selector, axis=0)
        else: used_neg_box_idx = np.array([], dtype=np.int32)

        if len(used_neg_box_idx) > 0:
            used_neg_box = np.take(box, used_neg_box_idx, axis=0)
        else: used_neg_box = np.zeros([0, 4], dtype=np.float32)
        used_box = np.concatenate([used_box, used_neg_box], axis=0)

        # [6] Add pad boxes (which covers whole image). this is to fix num_box
        box_idx_start['pad_box'] = used_box.shape[0]
        num_pad_box = MAX_USED_BOX - box_idx_start['pad_box']
        pad_box = np.tile(np.expand_dims(np.array(
            [0, 0, self.width, self.height], dtype=np.float32), axis=0),
            [num_pad_box, 1])
        used_box = np.concatenate([used_box, pad_box], axis=0)

        # [7] Construct region description batch - required for caption generation.
        # You can used the data in model with following way
        # 1. box selection, 2. get feature from box, 3. description, 4. loss
        all_desc = entry['region_descriptions'].value
        all_desc_len = entry['region_description_len'].value
        if len(asn_idx['region']) > 0:
            asn_desc = np.take(all_desc, asn_idx['region'], axis=0)
            asn_desc_len = np.take(all_desc_len, asn_idx['region'], axis=0)
        else:
            asn_desc = np.zeros([0, all_desc.shape[1]], dtype=np.int32)
            asn_desc_len = np.zeros([0], dtype=np.int32)
        asn_desc_box_idx = entry['asn_region2pos_idx'].value

        if len(used_no_asn_idx['region']) > 0:
            used_no_asn_desc = np.take(all_desc, used_no_asn_idx['region'], axis=0)
            used_no_asn_desc_len = np.take(
                all_desc_len, used_no_asn_idx['region'], axis=0)
        else:
            used_no_asn_desc = np.zeros([0, all_desc.shape[1]], dtype=np.int32)
            used_no_asn_desc_len = np.zeros([0], dtype=np.int32)
        used_no_asn_desc_box_idx = \
            np.arange(len(used_no_asn_idx['region']), dtype=np.int32) + \
            box_idx_start['region']

        used_desc = np.concatenate(
            [asn_desc, used_no_asn_desc], axis=0)
        used_desc_len = np.concatenate(
            [asn_desc_len, used_no_asn_desc_len], axis=0)
        used_desc_box_idx = np.concatenate(
            [asn_desc_box_idx, used_no_asn_desc_box_idx], axis=0)
        num_used_desc = used_desc.shape[0]

        # pad descriptions to have fixed size MAX_BOX_PER_ENTRY['region']
        pad_size = MAX_BOX_PER_ENTRY['region'] - num_used_desc
        if pad_size > 0:
            pad_desc = np.zeros([pad_size, used_desc.shape[1]], dtype=np.int32)
            pad_desc_len = np.zeros([pad_size], dtype=np.int32) + 1
            pad_desc_box_idx = np.zeros([pad_size], dtype=np.int32)
            used_desc = np.concatenate([used_desc, pad_desc], axis=0)
            used_desc_len = np.concatenate([used_desc_len, pad_desc_len], axis=0)
            used_desc_box_idx = np.concatenate([used_desc_box_idx,
                                                pad_desc_box_idx], axis=0)
        # add end token <e> to description
        used_desc = np.concatenate(
            [used_desc, np.zeros([used_desc.shape[0], 1], dtype=np.int32)],
            axis=1)
        for i in range(used_desc.shape[0]):
            used_desc[i, used_desc_len[i]] = self.vocab['dict']['<e>']
        # for using end token, use used_desc_len + 1 as the length

        # blank-fill
        blank_desc = []
        blank_desc_len = []
        for i in range(used_desc.shape[0]):
            drop_count = RANDOM_STATE.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            drop_count = min(drop_count, used_desc_len[i])
            # drop start and end
            drop_s = RANDOM_STATE.randint(0, used_desc_len[i] - drop_count + 1)
            drop_e = drop_s + drop_count
            a_blank_desc = np.zeros([used_desc.shape[1]], dtype=np.int32)
            a_blank_desc[:drop_s] = used_desc[i][:drop_s]
            a_blank_desc[drop_s] = self.vocab['dict']['<unk>']
            left_part = used_desc[i][drop_e:]
            a_blank_desc[drop_s + 1: drop_s + 1 + len(left_part)] = left_part
            a_blank_desc_len = used_desc_len[i] - drop_count + 1
            blank_desc.append(a_blank_desc)
            blank_desc_len.append(a_blank_desc_len)
        blank_desc = np.stack(blank_desc, axis=0)
        blank_desc_len = np.array(blank_desc_len, dtype=np.int32)

        # TODO(hyeonwoonoh): add mask for filtering padded descriptions

        # [8] additional information for image retrieval / language retrival with
        # description
        desc_idx_set = set(range(num_used_desc))  # do not use pad as gt
        # use all box except for padding
        box_idx_set = set(range(box_idx_start['pad_box']))
        lr_desc_idx = []  # lr stands for language retrieval
        lr_gt = []
        ir_box_idx = [] # ir stands for image retrieval
        ir_gt = []
        for i in range(used_desc.shape[0]):
            pos_desc_list = [i]
            pos_box_list = [used_desc_box_idx[i]]
            neg_desc_list = list(desc_idx_set - set(pos_desc_list))
            neg_box_list = list(box_idx_set - set(pos_box_list))

            RANDOM_STATE.shuffle(neg_desc_list)
            RANDOM_STATE.shuffle(neg_box_list)

            sampled_desc_idx = \
                (pos_desc_list + neg_desc_list)[: LANGUAGE_RETRIEVAL_K]
            sampled_box_idx = \
                (pos_box_list + neg_box_list)[: IMAGE_RETRIEVAL_K]
            gt_desc = np.zeros([LANGUAGE_RETRIEVAL_K], dtype=np.float32)
            gt_desc[0] = 1
            gt_box = np.zeros([IMAGE_RETRIEVAL_K], dtype=np.float32)
            gt_box[0] = 1

            lr_desc_idx.append(sampled_desc_idx)
            lr_gt.append(gt_desc)
            ir_box_idx.append(sampled_box_idx)
            ir_gt.append(gt_box)
        lr_desc_idx = np.stack(lr_desc_idx, axis=0)
        lr_gt = np.stack(lr_gt, axis=0)
        ir_box_idx = np.stack(ir_box_idx, axis=0)
        ir_gt = np.stack(ir_gt, axis=0)

        # [9] Data for classification of object / attribute / relationship
        entry_candidate = {}
        entry_candidate_len = {}
        entry_candidate_name = {}
        entry_selection_gt = {}
        used_entry_box_idx = {}
        num_used_entry = {}
        for key in ['object', 'attribute', 'relationship']:
            intseq_candidate_idx = []
            all_name_ids = entry['{}_name_ids'.format(key)].value
            # TODO(hyeonwoonoh): change "{}_names" to "{}_num_names" for new
            # merged dataset
            all_num_names = entry['{}_num_names'.format(key)].value
            if len(asn_idx[key]) > 0:
                asn_name_ids = np.take(all_name_ids, asn_idx[key], axis=0)
                asn_num_names = np.take(all_num_names, asn_idx[key], axis=0)
            else:
                asn_name_ids = np.zeros([0, all_name_ids.shape[1]], dtype=np.int32)
                asn_num_names = np.zeros([0], dtype=np.int32)
            asn_entry_box_idx = entry['asn_{}2pos_idx'.format(key)].value

            if len(used_no_asn_idx[key]) > 0:
                used_no_asn_name_ids = np.take(
                    all_name_ids, used_no_asn_idx[key], axis=0)
                used_no_asn_num_names = np.take(
                    all_num_names, used_no_asn_idx[key], axis=0)
            else:
                used_no_asn_name_ids = np.zeros(
                    [0, all_name_ids.shape[1]], dtype=np.int32)
                used_no_asn_num_names = np.zeros([0], dtype=np.int32)
            used_no_asn_entry_box_idx = \
                np.arange(len(used_no_asn_idx[key]), dtype=np.int32) + \
                box_idx_start[key]

            used_name_ids = np.concatenate(
                [asn_name_ids, used_no_asn_name_ids], axis=0)
            used_num_names = np.concatenate(
                [asn_num_names, used_no_asn_num_names], axis=0)
            used_entry_box_idx[key] = np.concatenate(
                [asn_entry_box_idx, used_no_asn_entry_box_idx], axis=0)
            num_used_entry[key] = used_name_ids.shape[0]

            pad_size = MAX_BOX_PER_ENTRY[key] - num_used_entry[key]
            if pad_size > 0:
                pad_name_ids = np.zeros(
                    [pad_size, used_name_ids.shape[1]], dtype=np.int32)
                pad_num_names = np.zeros([pad_size], dtype=np.int32)
                pad_entry_box_idx = np.zeros([pad_size], dtype=np.int32)
                used_name_ids = np.concatenate(
                    [used_name_ids, pad_name_ids], axis=0)
                used_num_names = np.concatenate(
                    [used_num_names, pad_num_names], axis=0)
                used_entry_box_idx[key] = np.concatenate(
                    [used_entry_box_idx[key], pad_entry_box_idx], axis=0)

            total_name_ids_set = set(list(range(self.num_entry[key])))
            entry_candidate[key] = []
            entry_candidate_len[key] = []
            entry_candidate_name[key] = []
            entry_selection_gt[key] = []
            for i in range(used_name_ids.shape[0]):
                a_name_ids = list(used_name_ids[i])
                a_num_names = int(used_num_names[i])

                a_neg_ids = list(total_name_ids_set - set(a_name_ids))
                RANDOM_STATE.shuffle(a_neg_ids)

                a_sampled_ids = (a_name_ids + a_neg_ids)[: NUM_K]
                a_sampled_entry = np.take(
                    self.entry[key], a_sampled_ids, axis=0)
                a_sampled_entry_len = np.take(
                    self.entry_len[key], a_sampled_ids, axis=0)
                a_sampled_entry_name = np.take(
                    self.entry_name[key], a_sampled_ids, axis=0)
                a_sampled_entry_gt = np.zeros([NUM_K], dtype=np.float32)
                a_sampled_entry_gt[: a_num_names] = 1

                entry_candidate[key].append(a_sampled_entry)
                entry_candidate_len[key].append(a_sampled_entry_len)
                entry_candidate_name[key].append(a_sampled_entry_name)
                entry_selection_gt[key].append(a_sampled_entry_gt)
            entry_candidate[key] = np.stack(entry_candidate[key], axis=0)
            entry_candidate_len[key] = np.stack(entry_candidate_len[key], axis=0)
            entry_candidate_name[key] = np.stack(entry_candidate_name[key], axis=0)
            entry_selection_gt[key] = np.stack(entry_selection_gt[key], axis=0)
        """
        Returns:
            * image and box:
                - image: resized rgb image (scale: [0, 255])
                - box: all set of boxes used for roi pooling (x1y1x2y2 format)
            * description:
                - desc: region description ground truths
                - desc_len: region description lengths
                - desc_box_idx: index of description corresponded boxes
                - num_used_desc: number of descriptions except for padding
            * blank fill:
                - blank_desc: region description with random blank
                - blank_desc_len: length of blank_desc
            * language retrieval (lr) and image retrieval (ir):
                - lr_desc_idx: description candidates index for matching
                - lr_gt: ground truth retrieval results (one-hot)
                - ir_box_idx: candidate boxes index
                - ir_gt: ground truth retrieval results (one-hot)
            * entry [object, attribute, relationship] classification:
                - {}_box_idx: box index for entry classification
                - {}_num_used_box: number of used boxes (for masking loss)
                - {}_candidate: intseq of candidate entry names
                - {}_candidate_len: intseq length  of candidate entry names
                - {}_candidate_name: string of candidate entry (for debugging)
                - {}_selection_gt: gt for entry selection task
        """
        returns = {
            'image': image,
            'box': used_box,
            'desc': used_desc,
            'desc_len': used_desc_len,
            'desc_box_idx': used_desc_box_idx,
            'num_used_desc': num_used_desc,
            'blank_desc': blank_desc,
            'blank_desc_len': blank_desc_len,
            'lr_desc_idx': lr_desc_idx,
            'lr_gt': lr_gt,
            'ir_box_idx': ir_box_idx,
            'ir_gt': ir_gt}
        for key in ['object', 'attribute', 'relationship']:
            returns['{}_box_idx'.format(key)] = used_entry_box_idx
            returns['{}_num_used_box'.format(key)] = num_used_entry[key]
            returns['{}_candidate'.format(key)] = entry_candidate[key]
            returns['{}_candidate_len'.format(key)] = entry_candidate_len[key]
            returns['{}_candidate_name'.format(key)] = entry_candidate_name[key]
            returns['{}_selection_gt'.format(key)] = entry_selection_gt[key]

        return returns

    def get_data_shapes(self):
        data_shapes = {
            'image': [self.height, self.width, 3],
            'box': [MAX_USED_BOX, 4],
            'desc': [MAX_BOX_PER_ENTRY['region'], None],
            'desc_len': [MAX_BOX_PER_ENTRY['region']],
            'desc_box_idx': [MAX_BOX_PER_ENTRY['region']],
            'num_used_desc': (),
            'blank_desc': [MAX_BOX_PER_ENTRY['region'], None],
            'blank_desc_len': [MAX_BOX_PER_ENTRY['region']],
            'lr_desc_idx': [MAX_BOX_PER_ENTRY['region'], LANGUAGE_RETRIEVAL_K],
            'lr_gt': [MAX_BOX_PER_ENTRY['region'], LANGUAGE_RETRIEVAL_K],
            'ir_box_idx': [MAX_BOX_PER_ENTRY['region'], IMAGE_RETRIEVAL_K],
            'ir_gt': [MAX_BOX_PER_ENTRY['region'], IMAGE_RETRIEVAL_K],
        }
        for key in ['object', 'attribute', 'relationship']:
            data_shapes['{}_box_idx'.format(key)] = [MAX_BOX_PER_ENTRY[key]]
            data_shapes['{}_num_used_box'.format(key)] = ()
            data_shapes['{}_candidate'.format(key)] = \
                [MAX_BOX_PER_ENTRY[key], NUM_K, None]
            data_shapes['{}_candidate_len'.format(key)] = \
                [MAX_BOX_PER_ENTRY[key], NUM_K]
            data_shapes['{}_candidate_name'.format(key)] = \
                [MAX_BOX_PER_ENTRY[key], NUM_K]
            data_shapes['{}_selection_gt'.format(key)] = \
                [MAX_BOX_PER_ENTRY[key], NUM_K]

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
