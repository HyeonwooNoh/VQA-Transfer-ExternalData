import argparse
import h5py
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
        else:
            raise ValueError('Unknown model_type')
        return Model

    def __init__(self, config, dataset):
        self.config = config
        self.data_cfg = dataset.get_config()

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
        Model = self.get_model_class()
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
        data_info['max_box_num'] = np.array(
            self.data_cfg.max_roi_num, dtype=np.int32)
        for it in tqdm(range(self.num_iter), desc='extract feature'):
            try:
                res = self.session.run(feed_dict)
            except tf.errors.OutOfRangeError:
                log.warn('OutOfRangeError happens at {} iter'.format(it + 1))
            else:
                for b in range(res['vfeat'].shape[0]):
                    image_id = ''.join(
                        res['image_id'][b, :res['image_id_len'][b]])
                    num_box = res['num_box'][b]
                    vfeat = res['vfeat'][b, :num_box]
                    box = res['box'][b, :num_box]
                    normal_box = res['normal_box'][b, :num_box]

                    grp = f.create_group(image_id)
                    grp['num_box'] = num_box
                    grp['vfeat'] = vfeat
                    grp['box'] = box
                    grp['normal_box'] = normal_box

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
    parser.add_argument('--qa_split_dir', type=str,
                        default='data/preprocessed/vqa_v2'
                        '/qa_split_thres1_500_thres2_50', help=' ')
    parser.add_argument('--used_image_name', type=str,
                        default='used_image_path.txt', help=' ')
    parser.add_argument('--save_name', type=str,
                        default='used_vfeat.hdf5', help=' ')
    parser.add_argument('--image_dir', type=str,
                        default='data/VQA_v2/images', help=' ')
    parser.add_argument('--densecap_dir', type=str,
                        default='data/VQA_v2/densecap', help=' ')
    # hyper parameters
    parser.add_argument('--pretrained_param_path', type=str, default=None,
                        required=True)
    # model parameters
    parser.add_argument('--batch_size', type=int, default=96, help=' ')

    config = parser.parse_args()
    config.used_image_path = os.path.join(config.qa_split_dir,
                                          config.used_image_name)
    config.save_path = os.path.join(config.qa_split_dir,
                                    config.save_name)
    check_config(config)

    dataset = dataset_vfeat.create_dataset(
        config.used_image_path, config.image_dir, config.densecap_dir,
        is_train=False)
    config.dataset_config = dataset.get_config()

    extractor = Extractor(config, dataset)
    extractor.extract()

if __name__ == '__main__':
    main()
