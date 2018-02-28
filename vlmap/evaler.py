import argparse
import os
import time
import numpy as np
import tensorflow as tf

from PIL import Image

from util import log
from vlmap.datasets import dataset_objects, input_ops_objects


class Evaler(object):

    @staticmethod
    def get_model_class(model_name='default'):
        if model_name == 'default':
            from model import Model
        return Model

    def __init__(self, config, object_datasets):
        self.config = config

        self.split = config.split
        self.train_dir = config.train_dir
        self.batch_size = config.batch_size
        self.max_steps = config.max_steps
        self.save_output = config.save_output

        # Input
        self.batches = {}
        with tf.name_scope('datasets/object_batch'):
            self.batches['object'] = input_ops_objects.create(
                object_datasets[self.split], self.batch_size,
                is_train=False, scope='input_ops', shuffle=False)

        # Model
        Model = self.get_model_class()
        log.infov('using model class: {}'.format(Model))
        self.model = Model(self.batches, config, is_train=False)

        self.global_step = tf.train.get_or_create_global_step(graph=None)

        # Checkpoint
        all_vars = tf.trainable_variables()
        enc_I_vars, learn_vars = self.model.filter_vars(all_vars)
        log.warn('Variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(learn_vars, print_info=True)

        tf.set_random_seed(123)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1})

        self.session = tf.Session(config=session_config)

        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint = config.checkpoint
        if self.checkpoint is None:
            self.checkpoint = tf.train.latest_checkpoint(self.train_dir)

    def eval_run(self):
        if self.checkpoint == '':
            raise ValueError('No checkpoint.')
        else:
            self.saver.restore(self.session, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start Inference and Evaluation")
        it = 0
        accumulated_report = {}
        while self.max_step == -1 or it < self.max_steps:
            step_result = self.run_eval_step()

            self.log_step_message(step_result)
            self.accumulate_report(self, step_result['report'],
                                   accumulated_report)
            if self.save_output:
                self.save_step_output(step_result)

        self.average_accumulated_report(accumulated_report)
        self.log_final_message(accumulated_report)

    def save_step_output(self, step_result):
        save_dir = os.path.join(self.checkpoint + '_output',
                                'step_{}'.format(step_result['step']))
        image_dir = os.path.join(save_dir, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        image_paths = []
        for i, image in enumerate(step_result['output']['image']):
            image_path = os.path.join(image_dir, '{:05d}.jpg'.format(i))
            pil_image = Image.fromarray(image.astype(np.uint8))
            pil_image.save(image_path)
            image_paths.append(image_path)
        with open(os.path.join(save_dir, 'index.html'), 'w') as f:
            for i, image_path in enumerate(image_paths):
                html = '<div><img src="{}"><p>{}</p></div>'.format(
                    image_path, step_result['prediction_string'][i])
                f.write(html + '\n')

    def accumulate_report(self, report, accumulated_report):
        if 'accum_count' in accumulated_report:
            for k in accumulated_report.keys():
                if k != 'accum_count':
                    accumulated_report[k] += report[k]
            accumulated_report['accum_count'] += 1
        else:
            accumulated_report = {}
            for k in report.keys():
                accumulated_report[k] = report[k]
            accumulated_report['accum_count'] = 1

    def average_accumulated_report(self, accumulated_report):
        accum_count = accumulated_report['accum_count']
        for k in accumulated_report.keys():
            accumulated_report[k] /= float(accum_count)
        del accumulated_report['accum_count']
        return accumulated_report

    def run_eval_step(self):
        _start_time = time.time()
        fetch = {
            'step': self.global_step,
            'report': self.model.report,
            'output': self.model.output,
        }
        fetch_values = self.session.run(fetch)
        _end_time = time.time()
        return fetch_values, (_end_time - _start_time)

    def log_step_message(self, step_result):
        report_msg = ' '.join(['{}: {}'.format(k, step_result['report'][k])
                               for k in sorted(step_result['report'].keys())])
        msg = '[{}] step {}: [{}]'.format(
            self.split, step_result['step'], report_msg)
        log.info(msg)
        return msg

    def log_final_message(self, final_report):
        report_msg = ' '.join(['{}: {}'.format(k, final_report[k])
                               for k in sorted(final_report.keys())])
        msg = '[{}] Final average report: [{}]'.format(
            self.split, report_msg)
        log.info(msg)
        with open(self.checkpoint + '.final.txt', 'r') as f:
            f.write(msg)
        return msg


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # paths
    parser.add_argument('--vocab_path', type=str,
                        default='data/preprocessed/vocab.json', help='')
    parser.add_argument('--image_dir', type=str,
                        default='data/VisualGenome/VG_100K', help='')
    parser.add_argument('--object_dataset_path', type=str,
                        default='data/preprocessed/objects_min_occ20', help='')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'val'], help=" ")
    # checkpoint
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    # hyper parameters
    parser.add_argument('--object_num_k', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='-1 for single epoch')
    parser.add_argument('--save_output', action='store_true', default=False)

    config = parser.parse_args()

    object_datasets = dataset_objects.create_default_splits(
        config.object_dataset_path, config.image_dir, config.vocab_path,
        config.object_num_k, is_train=False)
    config.object_data_shapes = \
        object_datasets[config.split].get_data_shapes()
    config.object_max_name_len = \
        object_datasets[config.split].max_name_len

    evaler = Evaler(config, object_datasets)
    evaler.eval_run()


if __name__ == '__main__':
    main()
