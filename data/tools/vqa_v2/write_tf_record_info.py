import argparse
import h5py
import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from util import log, tf_util

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--qa_split_dir', type=str,
                    default='data/preprocessed/vqa_v2'
                    '/qa_split_thres1_500_thres2_50', help=' ')
parser.add_argument('--dataset_name', type=str, default='data', help=' ')
parser.add_argument('--vfeat_name', type=str, default='used_vfeat.hdf5', help=' ')
parser.add_argument('--tf_record_dir', type=str, default='tf_record_wo_image',
                    help=' ')
config = parser.parse_args()

config.dataset_dir = os.path.join(config.qa_split_dir, config.dataset_name)
config.vfeat_path = os.path.join(config.qa_split_dir, config.vfeat_name)
config.tf_record_info_path = os.path.join(
    config.qa_split_dir, config.tf_record_dir, 'data_info.hdf5')

if os.path.exists(config.tf_record_info_path):
    raise ValueError('Do not overwrite: {}'.format(config.tf_record_info_path))

with h5py.File(os.path.join(dataset_dir, 'data.hdf5'), 'r') as data:
    max_q_len = data['data_info']['max_q_len'].value

with h5py.File(vfeat_path, 'r') as vfeat:
    max_box_num = vfeat['data_info']['max_box_num'].value
