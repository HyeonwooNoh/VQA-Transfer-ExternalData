import h5py
import os
import tensorflow as tf


def create(batch_size,
           tf_record_dir,
           split,
           is_train=True,
           scope='vqa_tf_record',
           shuffle=True):

    tf_record_info_path = os.path.join(tf_record_dir, 'data_info.hdf5')
    with h5py.File(tf_record_info_path, 'r') as f:
        max_q_len = int(f['data_info']['max_q_len'].value)
        max_box_num = int(f['data_info']['max_box_num'].value)
        vfeat_dim = int(f['data_info']['vfeat_dim'].value)

    tf_record_path = os.path.join(tf_record_dir, split, '{}-*'.format(split))
    with tf.device('/cpu:0'):
        files = tf.data.Dataset.list_files(tf_record_path)

        dataset = files.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=10, block_length=1))

        if is_train and shuffle:
            dataset = dataset.shuffle(buffer_size=3000)

        def parse_fn(example):
            example_fmt = {
                'qid': tf.FixedLenFeature((), tf.int64, -1),
                'image_id': tf.FixedLenFeature((), tf.string, ""),
                'box/list': tf.FixedLenFeature([max_box_num, 4], tf.float32),
                'box/shape': tf.FixedLenFeature([2], tf.int64),
                'num_box': tf.FixedLenFeature((), tf.int64, -1),
                'V_ft/list': tf.FixedLenFeature([max_box_num, vfeat_dim], tf.float32),
                'V_ft/shape': tf.FixedLenFeature([2], tf.int64),
                'q_intseq/list': tf.FixedLenFeature([max_q_len], tf.int64),
                'q_intseq/len': tf.FixedLenFeature((), tf.int64),
                'answer_id': tf.FixedLenFeature((), tf.int64),
            }
            parsed = tf.parse_single_example(example, example_fmt)

            inputs = {
                'id': parsed['qid'],
                'image_id': parsed['image_id'],
                'box': parsed['box/list'],
                'num_box': tf.cast(parsed['num_box'], tf.int32),
                'V_ft': parsed['V_ft/list'],
                'q_intseq': tf.cast(parsed['q_intseq/list'], tf.int32),
                'q_intseq_len': tf.cast(parsed['q_intseq/len'], tf.int32),
                'answer_id': tf.cast(parsed['answer_id'], tf.int32),
            }
            return inputs

        # TODO(hyeonwoonoh): fix map_and_batch.
        # Due to the bug in map_and_batch, we temporarily code batch_size by
        # placeholder_with_default.
        # Remove this when the tf bug is fixed.
        # ref: https://github.com/tensorflow/tensorflow/issues/17720
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=parse_fn,
            batch_size=tf.placeholder_with_default(
                tf.constant(batch_size, dtype=tf.int64), shape=())))

        dataset = dataset.prefetch(buffer_size=10)

        if is_train:
            dataset = dataset.repeat(1000)
        iterator = dataset.make_one_shot_iterator()
        batch_ops = iterator.get_next()

        return batch_ops
