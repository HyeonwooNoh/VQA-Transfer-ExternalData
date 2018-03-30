import argparse
import h5py
import json
import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from util import log
from vqa.datasets import dataset_vfeat, input_ops_vfeat


class Extractor(object):

    @staticmethod
    def get_model_class(model_type='vfeat'):
        if model_type == 'vfeat':
            from vqa.model_vfeat import Model
        elif model_type == 'resnet':
            from vqa.model_vfeat_resnet import Model
        else:
            raise ValueError('Unknown model_type')
        return Model

    def __init__(self, config, dataset):
        self.config = config
        self.data_cfg = dataset.get_config()

        self.image_path2idx = config.image_path2idx
        self.image_id2idx = config.image_id2idx

        self.save_path = config.save_path
        self.pretrained_param_path = config.pretrained_param_path

        log.infov('save_path: {}'.format(self.save_path))
        log.infov('pretrained_param_path: {}'.format(
            self.pretrained_param_path))

        self.batch_size = config.batch_size
        self.num_example = len(dataset.ids)
        self.num_iter = self.num_example / self.batch_size + 1

        with tf.name_scope('datasets/batch'):
            self.batch = input_ops_vfeat.create(
                dataset, self.batch_size, is_train=False, scope='batch_ops',
                shuffle=False)

        # Model
        Model = self.get_model_class(config.model_type)
        log.infov('using model class: {}'.format(Model))
        self.model = Model(self.batch, config, is_train=False)

        self.global_step = tf.train.get_or_create_global_step(graph=None)

        # Checkpoint and monitoring
        all_vars = tf.global_variables()
        log.warn('all variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        self.pretrain_loader = tf.train.Saver(var_list=all_vars, max_to_keep=1)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1})

        self.session = tf.Session(config=session_config)

        if self.pretrained_param_path is not None:
            log.info('Pre-trained param path: {}'.format(
                self.pretrained_param_path))
            self.pretrain_loader.restore(
                self.session, self.pretrained_param_path)
            log.info('Loaded the pre-trained parameters')

    def extract(self):
        log.infov('vfeat extraction start')

        feed_dict = {
            'vfeat': self.model.outputs['V_ft'],
            'box': self.batch['box'],
            'normal_box': self.batch['normal_box'],
            'num_box': self.batch['num_box'],
            'image_id': self.batch['image_id'],
            'image_id_len': self.batch['image_id_len']
        }

        f = h5py.File(self.save_path, 'w')
        data_info = f.create_group('data_info')
        data_info['pretrained_param_path'] = \
            self.pretrained_param_path.replace('/', '-')
        max_roi_num = self.data_cfg.max_roi_num
        data_info['max_box_num'] = np.array(
            self.data_cfg.max_roi_num, dtype=np.int32)

        vfeat_dim = 0
        initialized = False
        for it in tqdm(range(self.num_iter), desc='extract feature'):
            try:
                res = self.session.run(feed_dict)
            except tf.errors.OutOfRangeError:
                log.warn('OutOfRangeError happens at {} iter'.format(it + 1))
            else:
                for b in range(res['vfeat'].shape[0]):
                    image_id = ''.join(
                        res['image_id'][b, :res['image_id_len'][b]])
                    num_box = min(res['num_box'][b], max_roi_num)
                    vfeat = res['vfeat'][b, :num_box]
                    vfeat_dim = vfeat.shape[1]
                    box = res['box'][b, :num_box]
                    normal_box = res['normal_box'][b, :num_box]

                    if not initialized:
                        image_features = f.create_dataset(
                            'image_features',
                            (len(self.image_id2idx), max_roi_num, vfeat_dim), 'f')
                        normal_boxes = f.create_dataset(
                            'normal_boxes',
                            (len(self.image_id2idx), max_roi_num, 4), 'f')
                        num_boxes = np.zeros(
                            [len(self.image_id2idx)], dtype=np.int32) + num_box
                        spatial_features = f.create_dataset(
                            'spatial_features',
                            (len(self.image_id2idx), max_roi_num, 6), 'f')
                        initialized = True

                    image_idx = self.image_id2idx[image_id]
                    image_features[image_idx, :num_box, :] = vfeat  # add to hdf5
                    normal_boxes[image_idx, :num_box, :] = normal_box

                    ft_x1 = normal_box[:, 0]
                    ft_y1 = normal_box[:, 1]
                    ft_x2 = normal_box[:, 2]
                    ft_y2 = normal_box[:, 3]
                    ft_w = ft_x2 - ft_x1
                    ft_h = ft_y2 - ft_y1
                    spatial_features[image_idx, :num_box, :] = np.stack(
                        [ft_x1, ft_y1, ft_x2, ft_y2, ft_w, ft_h], axis=1)

        f['num_boxes'] = num_boxes

        data_info['vfeat_dim'] = np.array(vfeat_dim, dtype=np.int32)
        log.infov('iteration terminated at [{}/{}] iter'.format(
            it + 1, self.num_iter))
        f.close()
        log.warn('vfeat extraction is done: {}'.format(self.save_path))


def check_config(config):
    if os.path.exists(config.save_path):
        raise ValueError(
            'specified save_path exists already. do not overwrite: {}'.format(
                config.save_path))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # paths
    parser.add_argument('--tf_record_memft_dir', type=str,
                        default='data/preprocessed/vqa_v2'
                        '/new_qa_split_thres1_500_thres2_50/tf_record_memft',
                        help=' ')
    parser.add_argument('--save_name', type=str,
                        default='vfeat_extracted.hdf5', help=' ')
    parser.add_argument('--image_dir', type=str,
                        default='data/VQA_v2/images', help=' ')
    parser.add_argument('--densecap_dir', type=str,
                        default='data/VQA_v2/densecap', help=' ')
    # hyper parameters
    parser.add_argument('--pretrained_param_path', type=str, default=None,
                        required=True)
    # model parameters
    parser.add_argument('--batch_size', type=int, default=96, help=' ')
    parser.add_argument('--model_type', type=str, default='vfeat', help=' ',
                        choices=['vfeat', 'resnet'])

    config = parser.parse_args()

    config.image_info_path = os.path.join(config.tf_record_memft_dir,
                                          'image_info.json')

    config.save_path = os.path.join(config.tf_record_memft_dir,
                                    config.save_name)
    check_config(config)


    log.infov('loading image_info: {}'.format(config.image_info_path))
    image_info = json.load(open(config.image_info_path, 'r'))
    config.image_id2idx = image_info['image_id2idx']
    config.image_path2idx = image_info['image_path2idx']
    config.image_num2path = image_info['image_num2path']
    log.infov('done')

    dataset = dataset_vfeat.create_dataset(
        config.image_path2idx.keys(), config.image_dir, config.densecap_dir,
        is_train=False)
    config.dataset_config = dataset.get_config()

    extractor = Extractor(config, dataset)
    extractor.extract()

if __name__ == '__main__':
    main()
