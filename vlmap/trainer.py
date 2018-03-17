import argparse
import os
import time
import tensorflow as tf

from util import log
from vlmap.datasets import dataset_vlmap, input_ops_vlmap


class Trainer(object):

    @staticmethod
    def get_model_class(model_type='vlmap'):
        if model_type == 'vlmap':
            from model_vlmap import Model
        elif model_type == 'vljoint':
            from model_vljoint import Model
        return Model

    def __init__(self, config, dataset):
        self.config = config

        dataset_str = 'd'
        dataset_str += '_' + '_'.join(config.dataset_path.replace(
            'data/preprocessed/visualgenome/', '').split('/'))

        hyper_parameter_str = 'bs{}_lr{}_declw{}'.format(
            config.batch_size, config.learning_rate,
            config.decoder_loss_weight)

        if config.ft_enc_I:
            hyper_parameter_str += '_ft_enc_I'
        if config.decoder_type != 'glove_et':
            hyper_parameter_str += '_{}'.format(config.decoder_type)
        if config.description_task != 'blank-fill':
            hyper_parameter_str += '_{}'.format(config.description_task)
        if config.no_V_grad_enc_L:
            hyper_parameter_str += '_no_V_grad_enc_L'
        if config.use_relation:
            hyper_parameter_str += '_use_relation'
        if config.num_aug_retrieval > 0:
            hyper_parameter_str += '_aug_retriev{}'.format(
                config.num_aug_retrieval)

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
            vlmap_batch = {
                'train': input_ops_vlmap.create(
                    dataset['train'], self.batch_size, is_train=True,
                    scope='train_ops', shuffle=True),
                'val': input_ops_vlmap.create(
                    dataset['val'], self.batch_size, is_train=True,
                    scope='val_ops', shuffle=False)}
            self.batch = tf.case(
                {tf.equal(self.target_split, 'train'): lambda: vlmap_batch['train'],
                 tf.equal(self.target_split, 'val'): lambda: vlmap_batch['val']},
                default=lambda: vlmap_batch['train'], exclusive=True)

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

        if self.model.l_loss != 0:
            self.l_optimizer = tf.contrib.layers.optimize_loss(
                loss=self.model.l_loss,
                global_step=self.global_step,
                learning_rate=self.learning_rate,
                optimizer=tf.train.AdamOptimizer,
                clip_gradients=20.0,
                variables=learn_l_vars,
                increment_global_step=False,
                name='l_optimizer')
        else:
            self.l_optimizer = tf.no_op()

        self.summary_ops = {
            'train': tf.summary.merge_all(key='train'),
            'val': tf.summary.merge_all(key='val'),
            'heavy_train': tf.summary.merge_all(key='heavy_train'),
            'heavy_val': tf.summary.merge_all(key='heavy_val')
        }

        self.saver = tf.train.Saver(max_to_keep=100)
        self.enc_I_saver = tf.train.Saver(var_list=enc_I_vars, max_to_keep=1)
        self.pretrain_saver = tf.train.Saver(max_to_keep=1)
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
                self.run_train_step(s % self.heavy_summary_step == 0)
            if s % self.log_step == 0:
                self.log_step_message(step, loss, step_time, is_train=True)

            # Periodic inference
            if s % self.val_sample_step == 0:
                val_step, val_summary, val_loss, val_step_time = \
                    self.run_val_step(s % self.heavy_summary_step == 0)
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

    def run_train_step(self, use_heavy_summary):
        if use_heavy_summary: summary_key = 'heavy_train'
        else: summary_key = 'train'
        _start_time = time.time()
        fetch = [self.global_step, self.summary_ops[summary_key],
                 self.model.loss, self.v_optimizer, self.l_optimizer]
        fetch_values = self.session.run(fetch,
                                        feed_dict={self.target_split: 'train'})
        [step, summary, loss] = fetch_values[:3]
        _end_time = time.time()
        return step, summary, loss, (_end_time - _start_time)

    def run_val_step(self, use_heavy_summary):
        if use_heavy_summary: summary_key = 'heavy_val'
        else: summary_key = 'val'
        _start_time = time.time()
        fetch = [self.global_step, self.summary_ops['heavy_val'],
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


def check_config(config):
    if config.description_task != 'blank-fill' and config.no_V_grad_enc_L:
        raise ValueError('Set no_V_grad_enc_L only for blank-fill task')
    if config.num_aug_retrieval >= config.batch_size:
        raise ValueError('Set num_aug_retrieval smaller than batch_size')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # paths
    parser.add_argument('--vocab_path', type=str,
                        default='data/preprocessed/new_vocab50.json', help=' ')
    parser.add_argument('--glove_path', type=str,
                        default='data/preprocessed/glove.new_vocab50.300d.hdf5',
                        help=' ')
    parser.add_argument('--image_dir', type=str,
                        default='data/VisualGenome/VG_100K', help=' ')
    parser.add_argument('--dataset_path', type=str,
                        default='data/preprocessed/visualgenome'
                        '/merged_by_image_new_vocab50_min_region20', help=' ')
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--heavy_summary_step', type=int, default=1000)
    parser.add_argument('--val_sample_step', type=int, default=100)
    parser.add_argument('--write_summary_step', type=int, default=100)
    # hyper parameters
    parser.add_argument('--prefix', type=str, default='default', help=' ')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001, help=' ')
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    # model parameters
    parser.add_argument('--batch_size', type=int, default=3, help=' ')
    # vlmap: separate vision and language mapping
    # vljoint: having vision-language joint embedding space and learn v2j, l2j
    parser.add_argument('--model_type', type=str, default='vlmap', help=' ',
                        choices=['vlmap', 'vljoint'])
    parser.add_argument('--ft_enc_I', action='store_true', default=False)
    parser.add_argument('--use_relation', action='store_true', default=False)
    # glove_et: glove with embedding transform
    # glove: direct matching with glove vector
    # dense: not using glove vector, but using dense prediction layer
    parser.add_argument('--decoder_type', type=str, default='glove_et',
                        choices=['glove_et', 'glove', 'dense',
                                 'dense_n_le'], help=' ')
    # generation: generating description using visual feature directlry
    # blank-fill: using visual feature only for filling blanks
    parser.add_argument('--description_task', type=str, default='blank-fill',
                        choices=['generation', 'blank-fill'], help=' ')
    parser.add_argument('--decoder_loss_weight', type=float, default=1.0,
                        help=' ')
    parser.add_argument('--no_V_grad_enc_L', action='store_true', default=False,
                        help=' ')  # only for description_task == blank-fill
    parser.add_argument('--num_aug_retrieval', type=int, default=2,
                        help='Augment retrieval with interbatch data')

    config = parser.parse_args()
    check_config(config)

    dataset = dataset_vlmap.create_default_splits(
        config.dataset_path, config.image_dir, config.vocab_path,
        is_train=True)
    config.dataset_config = dataset['train'].get_config()

    trainer = Trainer(config, dataset)
    trainer.train()

if __name__ == '__main__':
    main()
