import argparse
import os
import time
import tensorflow as tf

from util import log
from vlmap.datasets import dataset_objects, input_ops_objects
from vlmap.datasets import dataset_region_descriptions, \
    input_ops_region_descriptions


class Trainer(object):

    @staticmethod
    def get_model_class(model_name='default'):
        if model_name == 'default':
            from model import Model
        return Model

    def __init__(self, config, datasets):
        self.config = config

        dataset_str = 'd'
        if not config.no_object:
            dataset_str += '_' + '_'.join(config.object_dataset_path.replace(
                'data/preprocessed/', '').split('/'))
        if not config.no_region:
            dataset_str += '_' + '_'.join(config.region_dataset_path.replace(
                'data/preprocessed/', '').split('/'))

        hyper_parameter_str = 'lr{}'.format(config.learning_rate)
        if not config.no_object:
            hyper_parameter_str += '_objbs{}'.format(config.object_batch_size)
        if not config.no_region:
            hyper_parameter_str += '_regbs{}'.format(config.region_batch_size)

        if config.no_object:
            hyper_parameter_str += '_no_obj'
        if config.no_region:
            hyper_parameter_str += '_no_reg'

        if config.finetune_enc_I:
            hyper_parameter_str += '_ft_enc_I'
        if config.no_V_grad_enc_L:
            hyper_parameter_str += '_no_V_grad_enc_L'
        if config.no_V_grad_dec_L:
            hyper_parameter_str += '_no_V_grad_dec_L'
        if config.no_L_grad_dec_L:
            hyper_parameter_str += '_no_L_grad_dec_L'

        self.train_dir = './train_dir/{}_{}_{}_{}'.format(
            dataset_str, config.prefix, hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # Input
        self.batch_size = 0
        if not config.no_object:
            self.batch_size += config.object_batch_size
        if not config.no_region:
            self.batch_size += config.region_batch_size
        self.object_batch_size = config.object_batch_size
        self.region_batch_size = config.region_batch_size
        with tf.name_scope('datasets'):
            self.target_split = tf.placeholder(tf.string)
        self.batches = {}
        with tf.name_scope('datasets/object_batch'):
            object_batches = {
                'train': input_ops_objects.create(
                    datasets['object']['train'], self.object_batch_size,
                    is_train=True, scope='train_ops', shuffle=True),
                'val': input_ops_objects.create(
                    datasets['object']['val'], self.object_batch_size,
                    is_train=True, scope='val_ops', shuffle=False)}
            self.batches['object'] = tf.case(
                {tf.equal(self.target_split, 'train'): lambda: object_batches['train'],
                 tf.equal(self.target_split, 'val'): lambda: object_batches['val']},
                default=lambda: object_batches['train'], exclusive=True)

        with tf.name_scope('datasets/region_batch'):
            region_batches = {
                'train': input_ops_region_descriptions.create(
                    datasets['region']['train'], self.region_batch_size,
                    is_train=True, scope='train_ops', shuffle=True),
                'val': input_ops_region_descriptions.create(
                    datasets['region']['val'], self.region_batch_size,
                    is_train=True, scope='val_ops', shuffle=False)}
            self.batches['region'] = tf.case(
                {tf.equal(self.target_split, 'train'): lambda: region_batches['train'],
                 tf.equal(self.target_split, 'val'): lambda: region_batches['val']},
                default=lambda: region_batches['train'], exclusive=True)

        # Model
        Model = self.get_model_class()
        log.infov('using model class: {}'.format(Model))
        self.model = Model(self.batches, config, is_train=True)

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
        all_vars = tf.trainable_variables()
        enc_I_vars, learn_v_vars, learn_l_vars = self.model.filter_vars(all_vars)
        log.warn('Trainable V variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(learn_v_vars, print_info=True)
        log.warn('Trainable L variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(learn_l_vars, print_info=True)

        self.v_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.v_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            variables=learn_v_vars,
            increment_global_step=True,
            name='v_optimizer')

        self.l_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.l_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            variables=learn_l_vars,
            increment_global_step=False,
            name='l_optimizer')

        self.summary_ops = {
            'train': tf.summary.merge_all(key='train'),
            'val': tf.summary.merge_all(key='val')
        }

        self.saver = tf.train.Saver(max_to_keep=100)
        self.enc_I_saver = tf.train.Saver(var_list=enc_I_vars, max_to_keep=1)
        self.pretrain_saver = tf.train.Saver(max_to_keep=1)
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

        enc_I_param_path = self.model.get_enc_I_param_path()
        if enc_I_param_path is not None:
            log.info('Enc_I parameter path: {}'.format(enc_I_param_path))
            self.enc_I_saver.restore(self.session, enc_I_param_path)
            log.info('Loaded pretrained Enc_I parameters')

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info('Checkpoint path: {}'.format(self.ckpt_path))
            self.pretrain_saver.restore(self.session, self.ckpt_path)
            log.info('Loaded the pretrained parameters')

    def train(self):
        log.infov('Training starts')

        max_steps = 1000000
        ckpt_save_steps = 5000

        for s in range(max_steps):
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
                 self.model.loss, self.v_optimizer, self.l_optimizer]
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
    parser.add_argument('--vocab_path', type=str,
                        default='data/preprocessed/vocab50.json', help='')
    parser.add_argument('--image_dir', type=str,
                        default='data/VisualGenome/VG_100K', help='')
    parser.add_argument('--object_dataset_path', type=str,
                        default='data/preprocessed/objects_vocab50_min_occ20', help='')
    parser.add_argument('--region_dataset_path', type=str,
                        default='data/preprocessed/region_descriptions_vocab50', help='')
    parser.add_argument('--used_wordset_path', type=str,
                        default='data/preprocessed/vocab50_used_wordset.hdf5', help='')
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--val_sample_step', type=int, default=100)
    parser.add_argument('--write_summary_step', type=int, default=100)
    # hyper parameters
    parser.add_argument('--object_num_k', type=int, default=500)
    parser.add_argument('--prefix', type=str, default='default', help='')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    # model parameters
    parser.add_argument('--object_batch_size', type=int, default=8, help='')
    parser.add_argument('--region_batch_size', type=int, default=8, help='')
    parser.add_argument('--no_object', action='store_true', default=False)
    parser.add_argument('--no_region', action='store_true', default=False)
    parser.add_argument('--finetune_enc_I', action='store_true', default=False)
    parser.add_argument('--no_V_grad_enc_L', action='store_true', default=False)
    parser.add_argument('--no_V_grad_dec_L', action='store_true', default=False)
    parser.add_argument('--no_L_grad_dec_L', action='store_true', default=False)

    config = parser.parse_args()

    datasets = {}
    datasets['object'] = dataset_objects.create_default_splits(
        config.object_dataset_path, config.image_dir, config.vocab_path,
        config.object_num_k, is_train=True)
    config.object_data_shapes = datasets['object']['train'].get_data_shapes()
    config.object_max_name_len = datasets['object']['train'].max_name_len

    datasets['region'] = dataset_region_descriptions.create_default_splits(
        config.region_dataset_path, config.image_dir, config.vocab_path,
        config.used_wordset_path, is_train=True)
    config.region_data_shapes = datasets['region']['train'].get_data_shapes()
    config.region_max_len = datasets['region']['train'].max_len

    trainer = Trainer(config, datasets)
    trainer.train()

if __name__ == '__main__':
    main()
