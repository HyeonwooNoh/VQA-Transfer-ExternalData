import argparse
import glob
import h5py
import os
import numpy as np

import tensorflow as tf

from util import log
from vqa.evaler import Evaler


def check_config(config):
    if config.root_train_dir is None and len(config.train_dirs) == 0:
        raise ValueError('Set either root_train_dir or train_dirs')
    if config.root_train_dir is not None and len(config.train_dirs) > 0:
        raise ValueError('Do not set both root_train_dir and train_dirs')


def parse_checkpoint(config):
    config.ckpt_name = config.checkpoint.split('/')[-1]

    dirname = config.checkpoint.split('/')[-2]
    #config.model_type = dirname.split('vqa_')[1].split('_d_')[0]
    config.model_type = dirname[4:].split('_d_')[0]

    qa_split_name = dirname.split('_d_')[1].split('_tf_record_memft')[0]
    config.tf_record_dir = os.path.join(
        'data/preprocessed/vqa_v2', qa_split_name, 'tf_record_memft')

    if 'vfeat_bottomup_36_my' in dirname:
        config.vfeat_name = 'vfeat_bottomup_36_my.hdf5'
    else:
        config.vfeat_name = 'vfeat_bottomup_36.hdf5'

    config.vocab_path = os.path.join(config.tf_record_dir, config.vocab_name)
    config.vfeat_path = os.path.join(config.tf_record_dir, config.vfeat_name)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # paths
    parser.add_argument('--image_dir', type=str, default='data/VQA_v2/images',
                        help=' ')
    parser.add_argument('--vocab_name', type=str, default='vocab.pkl', help=' ')
    # evaluation setting
    parser.add_argument('--max_iter', type=int, default=-1, help=' ')
    parser.add_argument('--split', type=str, default='test', help=' ',
                        choices=['train', 'val', 'testval', 'test'])
    # hyper parameters
    parser.add_argument('--prefix', type=str, default='default', help=' ')
    parser.add_argument('--root_train_dir', type=str, default=None, help=' ')
    parser.add_argument('--train_dirs', nargs='+', type=str, default=[], help=' ')
    # model parameters
    parser.add_argument('--batch_size', type=int, default=512, help=' ')
    parser.add_argument('--dump_heavy_output', action='store_true', default=False,
                        help=' ')
    parser.add_argument('--debug', type=int, default=0, help='0: normal, 1: debug')
    config = parser.parse_args()
    check_config(config)

    if config.root_train_dir is None:
        all_train_dirs = config.train_dirs
    else:
        all_train_dirs = glob.glob(os.path.join(config.root_train_dir, 'vqa_*'))
    all_train_dirs = sorted(all_train_dirs)

    log.warn('all_train_dirs:')
    for i, train_dir in enumerate(all_train_dirs):
        log.infov('{:02d}: {}'.format(i, train_dir))

    # Initialization
    filtered_train_dirs = []
    for train_dir in all_train_dirs:
        checkpoints = glob.glob(os.path.join(train_dir, 'model-*.index'))
        if len(checkpoints) == 0:
            continue
        else:
            filtered_train_dirs.append(train_dir)

        checkpoints = sorted([(int(c.split('model-')[1].split('.index')[0]),
                            c.split('.index')[0]) for c in checkpoints],
                            key=lambda x: x[0])

    all_train_dirs = filtered_train_dirs

    config.checkpoint = checkpoints[0][1]
    parse_checkpoint(config)

    log.infov('loading image features...')
    image_features = {}
    with h5py.File(config.vfeat_path, 'r') as f:
        image_features['features'] = np.array(f.get('image_features'))
        log.infov('feature done')
        image_features['spatials'] = np.array(f.get('spatial_features'))
        log.infov('spatials done')
        image_features['normal_boxes'] = np.array(f.get('normal_boxes'))
        log.infov('normal_boxes done')
        image_features['num_boxes'] = np.array(f.get('num_boxes'))
        log.infov('num_boxes done')
        image_features['max_box_num'] = int(f['data_info']['max_box_num'].value)
        image_features['vfeat_dim'] = int(f['data_info']['vfeat_dim'].value)
    loaded_vfeat_path = config.vfeat_path
    log.infov('done')

    # Iteration
    for train_dir in all_train_dirs:
        checkpoints = glob.glob(os.path.join(train_dir, 'model-*.index'))
        checkpoints = sorted([(int(c.split('model-')[1].split('.index')[0]),
                               c.split('.index')[0]) for c in checkpoints],
                             key=lambda x: x[0])
        config.checkpoint = checkpoints[0][1]
        parse_checkpoint(config)
        if loaded_vfeat_path != config.vfeat_path:
            log.warn(
                'vfeat_path for this train_dir is different from the ' +
                'initialized one: {} vs {}'.format(
                    loaded_vfeat_path, config.vfeat_path))
            continue

        for i, (model_i, checkpoint) in enumerate(checkpoints):
            log.warn('evaluate model-{} [{}/{}]: {}'.format(
                model_i, i, len(checkpoints), checkpoint))
            config.checkpoint = checkpoint
            evaler = Evaler(config, image_features=image_features)
            evaler.eval()
            evaler.session.close()
            tf.reset_default_graph()
    log.warn('all evaluation is done')

if __name__ == '__main__':
    main()
