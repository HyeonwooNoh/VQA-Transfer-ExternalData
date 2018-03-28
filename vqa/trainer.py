import argparse
import os
import time
import tensorflow as tf

from util import log
from vqa.datasets import input_ops_vqa_tf_record_memft as input_ops_vqa


class Trainer(object):

    @staticmethod
    def get_model_class(model_type='vqa'):
        if model_type == 'vqa':
            from vqa.model_vqa import Model
        elif model_type == 'standard':
            from vqa.model_standard import Model
        else:
            raise ValueError('Unknown model_type')
        return Model

    def __init__(self, config):
        self.config = config
        self.vfeat_path = config.vfeat_path
        self.tf_record_dir = config.tf_record_dir

        dataset_str = 'd'
        dataset_str += '_' + '_'.join(config.qa_split_dir.replace(
            'data/preprocessed/vqa_v2/', '').split('/'))
        dataset_str += '_' + config.tf_record_dir_name
        dataset_str += '_' + config.vfeat_name.replace('.hdf5', '')

        hyper_parameter_str = 'bs{}_lr{}'.format(
            config.batch_size, config.learning_rate)

        if config.ft_vlmap:
            hyper_parameter_str += '_ft_vlmap'

        self.train_dir = './train_dir/{}_{}_{}_{}_{}'.format(
            config.model_type, dataset_str, config.prefix, hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # Input
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

        # Optimizer
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate')

        # Checkpoint and monitoring
        trainable_vars = tf.trainable_variables()
        train_vars = self.model.filter_train_vars(trainable_vars)
        log.warn('Trainable variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(trainable_vars, print_info=True)
        log.warn('Filtered train variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(train_vars, print_info=True)

        self.optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            variables=train_vars,
            increment_global_step=True,
            name='optimizer')

        self.summary_ops = {
            'train': tf.summary.merge_all(key='train'),
            'val': tf.summary.merge_all(key='val'),
            'testval': tf.summary.merge_all(key='testval'),
            'heavy_train': tf.summary.merge_all(key='heavy_train'),
            'heavy_val': tf.summary.merge_all(key='heavy_val'),
            'heavy_testval': tf.summary.merge_all(key='heavy_testval'),
        }

        all_vars = tf.global_variables()
        transfer_vars = self.model.filter_transfer_vars(all_vars)

        self.saver = tf.train.Saver(max_to_keep=100)
        self.checkpoint_loader = tf.train.Saver(max_to_keep=1)
        self.pretrain_loader = tf.train.Saver(var_list=transfer_vars, max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.log_step = self.config.log_step
        self.heavy_summary_step = self.config.heavy_summary_step
        self.val_sample_step = self.config.val_sample_step
        self.write_summary_step = self.config.write_summary_step

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=None,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1})

        self.session = self.supervisor.prepare_or_wait_for_session(
            config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info('Checkpoint path: {}'.format(self.ckpt_path))
            self.checkpoint_loader.restore(self.session, self.ckpt_path)
            log.info('Loaded the checkpoint')

        self.pretrained_param_path = config.pretrained_param_path
        if self.pretrained_param_path is not None:
            log.info('Pre-trained param path: {}'.format(self.pretrained_param_path))
            self.pretrain_loader.restore(self.session, self.pretrained_param_path)
            log.info('Loaded the pre-trained parameters')

    def train(self):
        log.infov('Training starts')

        max_steps = 1000000
        ckpt_save_steps = 5000

        for s in range(max_steps):
            step, train_summary, loss, step_time = \
                self.run_train_step(s % self.heavy_summary_step == 0)
            if s % self.log_step == 0:
                self.log_step_message(step, loss, step_time,
                                      split='train', is_train=True)

            # Periodic inference
            if s % self.val_sample_step == 0:
                val_step, val_summary, val_loss, val_step_time = \
                    self.run_val_step(s % self.heavy_summary_step == 0,
                                      target_split='val')
                self.summary_writer.add_summary(val_summary,
                                                global_step=val_step)
                self.log_step_message(val_step, val_loss, val_step_time,
                                      split='val', is_train=False)
                testval_step, testval_summary, testval_loss, testval_step_time = \
                    self.run_val_step(s % self.heavy_summary_step == 0,
                                      target_split='testval')
                self.summary_writer.add_summary(testval_summary,
                                                global_step=testval_step)
                self.log_step_message(testval_step, testval_loss, testval_step_time,
                                      split='testval', is_train=False)

            if s % self.write_summary_step == 0:
                self.summary_writer.add_summary(train_summary,
                                                global_step=step)

            if s % ckpt_save_steps == 0:
                log.infov('Saved checkpoint at {}'.format(step))
                self.saver.save(
                    self.session, os.path.join(self.train_dir, 'model'),
                    global_step=step)

    def run_train_step(self, use_heavy_summary):
        if use_heavy_summary: summary_key = 'heavy_train'
        else: summary_key = 'train'
        _start_time = time.time()
        fetch = [self.global_step, self.summary_ops[summary_key],
                 self.model.loss, self.optimizer]
        fetch_values = self.session.run(fetch,
                                        feed_dict={self.target_split: 'train'})
        [step, summary, loss] = fetch_values[:3]
        _end_time = time.time()
        return step, summary, loss, (_end_time - _start_time)

    def run_val_step(self, use_heavy_summary, target_split):
        if use_heavy_summary: summary_key = 'heavy_{}'.format(target_split)
        else: summary_key = target_split
        _start_time = time.time()
        fetch = [self.global_step, self.summary_ops[summary_key],
                 self.model.loss]
        fetch_values = self.session.run(
            fetch, feed_dict={self.target_split: target_split})
        [step, summary, loss] = fetch_values[:3]
        _end_time = time.time()
        return step, summary, loss, (_end_time - _start_time)

    def log_step_message(self, step, loss, step_time, split='train',
                         is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} " +
                "instances/sec) "
                ).format(split_mode=split,
                         step=step,
                         loss=loss,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )


def check_config(config):
    if config.checkpoint is not None and config.pretrained_param_path is not None:
        raise ValueError('Do not set both checkpoint and pretrained_param_path')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # paths
    parser.add_argument('--image_dir', type=str, default='data/VQA_v2/images',
                        help=' ')
    parser.add_argument('--qa_split_dir', type=str,
                        default='data/preprocessed/vqa_v2'
                        '/new_qa_split_thres1_500_thres2_50', help=' ')
    parser.add_argument('--tf_record_dir_name', type=str,
                        default='tf_record_memft', help=' ')
    parser.add_argument('--vfeat_name', type=str,
                        default='vfeat_bottomup_36.hdf5', help=' ')
    parser.add_argument('--vocab_name', type=str, default='vocab.json', help=' ')
    # log
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--heavy_summary_step', type=int, default=1000)
    parser.add_argument('--val_sample_step', type=int, default=100)
    parser.add_argument('--write_summary_step', type=int, default=100)
    # hyper parameters
    parser.add_argument('--prefix', type=str, default='default', help=' ')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pretrained_param_path', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001, help=' ')
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    # model parameters
    parser.add_argument('--batch_size', type=int, default=3, help=' ')
    parser.add_argument('--model_type', type=str, default='vqa', help=' ',
                        choices=['vqa', 'standard'])
    parser.add_argument('--ft_vlmap', action='store_true', default=False)
    config = parser.parse_args()
    config.vocab_path = os.path.join(config.qa_split_dir, config.vocab_name)
    config.tf_record_dir = os.path.join(config.qa_split_dir,
                                        config.tf_record_dir_name)
    config.vfeat_path = os.path.join(config.tf_record_dir, config.vfeat_name)
    check_config(config)

    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
