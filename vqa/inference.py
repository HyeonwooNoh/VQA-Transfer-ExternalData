import collections
import os
import tensorflow as tf

from util import log
from vqa import importer
from vqa.datasets import input_ops_vqa_tf_record_memft as input_ops_vqa


class Inference(object):

    @staticmethod
    def get_model_class(model_type='vqa'):
        return importer.get_model_class(model_type)

    def __init__(self, config):
        self.config = config
        self.vfeat_path = config.vfeat_path
        self.tf_record_dir = config.tf_record_dir

        self.batch_size = config.batch_size
        with tf.name_scope('datasets'):
            self.target_split = tf.placeholder(tf.string)

        with tf.name_scope('datasets/batch'):
            vqa_batch = {
                'train': input_ops_vqa.create(
                    self.batch_size, self.tf_record_dir, 'train',
                    is_train=True, scope='train_ops', shuffle=True),
                'val': input_ops_vqa.create(
                    self.batch_size, self.tf_record_dir, 'val',
                    is_train=True, scope='val_ops', shuffle=False),
                'testval': input_ops_vqa.create(
                    self.batch_size, self.tf_record_dir, 'testval',
                    is_train=True, scope='testval_ops', shuffle=False),
                'test': input_ops_vqa.create(
                    self.batch_size, self.tf_record_dir, 'test',
                    is_train=True, scope='test_ops', shuffle=False)
            }
            batch_opt = {
                tf.equal(self.target_split, 'train'): lambda: vqa_batch['train'],
                tf.equal(self.target_split, 'val'): lambda: vqa_batch['val'],
                tf.equal(self.target_split, 'testval'): lambda: vqa_batch['testval'],
                tf.equal(self.target_split, 'test'): lambda: vqa_batch['test'],
            }
            self.batch = tf.case(
                batch_opt, default=lambda: vqa_batch['train'], exclusive=True)

        # Model
        Model = self.get_model_class(config.model_type)
        log.infov('using model class: {}'.format(Model))
        self.model = Model(self.batch, config, is_train=True)

        self.checkpoint_loader = tf.train.Saver(max_to_keep=1)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1})

        self.session = tf.Session(config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info('Checkpoint path: {}'.format(self.ckpt_path))
            self.checkpoint_loader.restore(self.session, self.ckpt_path)
            log.info('Loaded the checkpoint')
        log.warn('Inference initialization is done')


def get_model_types():
    return importer.get_model_types()


def parse_checkpoint(config):
    config.ckpt_name = config.checkpoint.split('/')[-1]

    dirname = config.checkpoint.split('/')[-2]
    config.model_type = dirname.split('vqa_')[1].split('_d_')[0]

    qa_split_name = dirname.split('_d_')[1].split('_tf_record_memft')[0]
    config.tf_record_dir = os.path.join(
        'data/preprocessed/vqa_v2', qa_split_name, 'tf_record_memft')

    if 'vfeat_bottomup_36_my' in dirname:
        config.vfeat_name = 'vfeat_bottomup_36_my.hdf5'
    else:
        config.vfeat_name = 'vfeat_bottomup_36.hdf5'

    config.vocab_path = os.path.join(config.tf_record_dir, config.vocab_name)
    config.vfeat_path = os.path.join(config.tf_record_dir, config.vfeat_name)


def get_default_config():
    config = collections.namedtuple('config', [])
    config.image_dir = 'data/VQA_v2/images'
    config.vocab_name = 'vocab.pkl'
    config.checkpoint = None
    config.batch_size = 512
    return config


def get_inference(config=None):
    if config is None:
        config = get_default_config()
    return Inference(config)
