import argparse
import h5py
import os

from util import log

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

with h5py.File(os.path.join(config.dataset_dir, 'data.hdf5'), 'r') as data:
    max_q_len = data['data_info']['max_q_len'].value

with h5py.File(config.vfeat_path, 'r') as vfeat:
    max_box_num = vfeat['data_info']['max_box_num'].value
    vfeat_dim = vfeat['data_info']['vfeat_dim'].value

with h5py.File(config.tf_record_info_path, 'w') as f:
    data_info = f.create_group('data_info')
    data_info['max_q_len'] = max_q_len
    data_info['max_box_num'] = max_box_num
    data_info['vfeat_dim'] = vfeat_dim

log.warn('writing data_info to {} is done'.format(config.tf_record_info_path))
