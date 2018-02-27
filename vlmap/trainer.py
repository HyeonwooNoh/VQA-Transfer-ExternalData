import argparse
import os
import time
import tensorflow as tf

from pprint import pprint

from util import log
from vlmap.datasets import dataset_objects, input_ops_objects


class Trainer(object):

    @staticmethod
    def get_model_class(model_name='default'):
        if model_name == 'default':
            from model import Model
        return Model

    def __init__(self, config, object_datasets):
        self.config = config

        dataset_str = ''
        dataset_str += '_'.join(config.object_dataset_path.replace(
            'data/preprocessed/', '').split('/'))

        hyper_parameter_str = 'bs_{}_lr_{}'.format(
            config.batch_size, config.learning_rate)

        self.train_dir = './train_dir/{}_{}_{}_{}'.format(
            dataset_str, config.prefix, hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # Input
        self.batch_size = config.batch_size
        with tf.name_scope('datasets'):
            self.target_split = tf.placeholder(tf.string)
        self.batches = {}
        with tf.name_scope('datasets/object_batch'):
            object_batches = {
                'train': input_ops_objects.create(
                    object_datasets['train'], self.batch_size, is_training=True,
                    scope='train_ops', shuffle=True),
                'val': input_ops_objects.create(
                    object_datasets['val'], self.batch_size, is_training=True,
                    scope='val_ops', shuffle=False)}
            self.batches['object'] = tf.case(
                {tf.equal(self.target_split, 'train'): lambda: object_batches['train'],
                 tf.equal(self.target_split, 'val'): lambda: object_batches['val']},
                default=lambda: object_batches['train'], exclusive=True)

        # Optimizer
        self.global_step = tf.train.get_or_create_global_step(graph=None)

        # Model
        Model = self.get_model_class()
        log.infov('using model class: {}'.format(Model))
        self.model = Model(self.batches, config, global_step=self.global_step,
                           is_train=True)

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
        all_vars = tf.trainable_variables()
        log.warn('Trainable variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        self.optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            name='optimizer')

        self.summary_ops = {
            'train': tf.summary.merge_all(key='train'),
            'val': tf.summary.merge_all(key='val')
        }

        self.saver = tf.train.Saver(max_to_keep=100)
        self.pretrain_saver = tf.train.Saver(var_list=all_vars, max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.log_step = self.config.log_step
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
            self.pretrain_saver.restore(self.session, self.ckpt_path)
            log.info('Loaded the pretrained parameters')

    def train(self):
        log.infov('Training starts')

        max_steps = 1000000
        ckpt_save_steps = 5000

        for s in xrange(max_steps):
            step, train_summary, loss, step_time = \
                self.run_train_step()
            if s % self.log_step == 0:
                self.log_step_message(step, loss, step_time, is_train=True)

            # Periodic inference
            if s % self.val_sample_step == 0:
                val_step, val_summary, val_loss, val_step_time = \
                    self.run_val_step()
                self.summary_writer.add_summary(val_summary,
                                                global_step=val_step)
                self.log_step_message(val_step, val_loss, val_step_time,
                                      is_train=False)

            if s % self.write_summary_step == 0:
                self.summary_writer.add_summary(train_summary,
                                                global_step=step)

            if s % ckpt_save_steps == 0:
                log.infov('Saved checkpoint at {}'.format(step))
                self.saver.save(
                    self.session, os.path.join(self.train_dir, 'model'),
                    global_step=step)

    def run_train_step(self):
        _start_time = time.time()
        fetch = [self.global_step, self.summary_ops['train'],
                 self.model.loss, self.optimizer]
        fetch_values = self.session.run(fetch,
                                        feed_dict={self.target_split: 'train'})
        [step, summary, loss] = fetch_values[:3]
        _end_time = time.time()
        return step, summary, loss, (_end_time - _start_time)

    def run_val_step(self):
        _start_time = time.time()
        fetch = [self.global_step, self.summary_ops['val'],
                 self.model.loss]
        fetch_values = self.session.run(fetch,
                                        feed_dict={self.target_split: 'val'})
        [step, summary, loss] = fetch_values[:3]
        _end_time = time.time()
        return step, summary, loss, (_end_time - _start_time)


    def log_step_message(self, step, loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} " +
                "instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # paths
    parser.add_argument('--image_dir', type=str,
                        default='data/VisualGenome/VG_100K', help='')
    parser.add_argument('--object_dataset_path', type=str,
                        default='data/preprocessed/objects_min_occ20', help='')
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--val_sample_step', type=int, default=100)
    parser.add_argument('--write_summary_step', type=int, default=100)
    # hyper parameters
    parser.add_argument('--object_num_k', type=int, default=500)
    parser.add_argument('--prefix', type=str, default='default', help=" ")
    parser.add_argument('--batch_size', type=int, default=32, help=" ")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001, help=" ")
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)

    config = parser.parse_args()

    object_datasets = dataset_objects.create_default_splits(
        config.object_dataset_path, config.image_dir, config.object_num_k,
        is_train=True)
    config.object_data_shapes = object_datasets['train'].get_data_shapes()
    config.object_max_name_len = object_datasets['train'].max_name_len

    trainer = Trainer(config, object_datasets)
    trainer.train()

if __name__ == '__main__':
    main()
