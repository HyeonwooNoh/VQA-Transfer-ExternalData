"""
Add denscap box to merged dataset.

Run this script from root directory, rather than from ./data.
"""
import argparse
import h5py
import os
import numpy as np

from tqdm import tqdm

from util import box_utils


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--densecap_results', type=str,
                    default='data/VisualGenome/VG_100K_densecap_output'
                    '/results_original_size.hdf5', help=' ')
parser.add_argument('--merged_dataset_dir', type=str,
                    default='data/preprocessed/visualgenome'
                    '/merged_by_image_vocab50', help=' ')
config = parser.parse_args()

densecap = h5py.File(config.densecap_results, 'r')
ids = open(os.path.join(config.merged_dataset_dir, 'id.txt'),
           'r').read().splitlines()
f = h5py.File(os.path.join(config.merged_dataset_dir, 'data.hdf5'), 'r+')


for id in tqdm(ids, desc='adding boxes to dataset'):
    r = densecap[id]
    f[id]['box_xywh'] = r['boxes'].value
    f[id]['box_scores'] = r['scores'].value

    box_xywh = r['boxes']

    gt_xywh_dict = {
        'object': f[id]['object_xywh'].value,
        'attribute': f[id]['attribute_xywh'].value,
        'relationship': f[id]['relationship_xywh'].value,
        'region': f[id]['region_xywh'].value
    }
    gt_xywh = np.concatenate(gt_xywh_dict.values(), axis=0)

    # positive and negative box idx
    dense2gt_iou = box_utils.iou_matrix_xywh(box_xywh, gt_xywh)
    is_positive = np.any(dense2gt_iou >= 0.7, axis=1)
    positive_box_idx = [i for i, p in enumerate(is_positive) if p]
    negative_box_idx = [i for i, p in enumerate(is_positive) if not p]

    f[id]['positive_box_idx'] = positive_box_idx
    f[id]['negative_box_idx'] = negative_box_idx

    pos_box_xywh = np.take(box_xywh, positive_box_idx, axis=0)
    neg_box_xywh = np.take(box_xywh, negative_box_idx, axis=0)

    for key in ['region', 'object', 'attribute', 'relationship']:
        # 1. is 'key entry' assigned to positive box
        # 2. index of assigned 'key entry'
        # 3. index of unassigned 'key entry'
        # 4. positive box index of assigned 'key entry'
        entry_xywh = gt_xywh_dict[key]
        entry2pos_iou = box_utils.iou_matrix_xywh(entry_xywh, pos_box_xywh)
        is_entry_assigned = np.any(entry2pos_iou >= 0.7, axis=1)
        asn_entry_idx = np.array(
            [i for i, p in enumerate(is_entry_assigned) if p],
            dtype=np.int32)
        no_asn_entry_idx = np.array(
            [i for i, p in enumerate(is_entry_assigned) if not p],
            dtype=np.int32)
        f[id]['asn_{}_idx'.format(key)] = asn_entry_idx
        f[id]['no_asn_{}_idx'.format(key)] = no_asn_entry_idx
        if len(asn_entry_idx) > 0:
            asn_entry2pos_idx = np.argmax(
                np.take(entry2pos_iou, asn_entry_idx, axis=0),
                axis=1).astype(np.int32)
            f[id]['asn_{}2pos_idx'.format(key)] = asn_entry2pos_idx
        else:
            f[id]['asn_{}2pos_idx'.format(key)] = np.array([], dtype=np.int32)

        # 1. chosen positive box idx for entry classification
        #   (which caption this box is about?)
        # 2. best answer (caption) index for each chosen box
        pos2entry_iou = entry2pos_iou.transpose()
        is_pos_assigned = np.any(pos2entry_iou >= 0.7, axis=1)
        asn_pos_idx = [i for i, p in enumerate(is_pos_assigned) if p]
        f[id]['{}_asn_pos_idx'.format(key)] = asn_pos_idx
        if len(asn_pos_idx) > 0:
            asn_pos2entry_idx = np.argmax(
                np.take(pos2entry_iou, asn_pos_idx, axis=0),
                axis=1).astype(np.int32)
            f[id]['{}_asn_pos2{}_idx'.format(key, key)] = asn_pos2entry_idx
        else:
            f[id]['{}_asn_pos2{}_idx'.format(key, key)] = np.array([], dtype=np.int32)
