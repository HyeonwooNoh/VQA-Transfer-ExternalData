import numpy as np
import tensorflow as tf

from util import log


def np2tf_type(np_type):
    if np_type == np.str: return tf.string
    elif np_type == np.int32: return tf.int32
    elif np_type == np.float32: return tf.float32
    elif np_type == np.bool: return tf.bool
    else: raise ValueError('Unknown np_type')


def create(dataset,
           batch_size,
           is_train=False,
           scope='vfeat_input',
           shuffle=True):

    ids = dataset.ids
    log.info('input_ops {}: using {} IDs from dataset'.format(
        scope, len(ids)))

    key_list = [
        'id',
        'box',
        'normal_box',
        'num_box',
        'image_id',
        'image_id_len',
    ]

    data_shapes = dataset.get_data_shapes()
    data_shapes['id'] = []

    with tf.device('/cpu:0'), tf.name_scope(scope):
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(ids))

        if is_train and shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=3000)

        def load_fn(id):
            data = dataset.get_data(id)
            data['id'] = id
            return tuple([data[key] for key in key_list])

        def load_py_func(id):

            data_types = dataset.get_data_types()
            data_types['id'] = np.str

            type_out = [np2tf_type(data_types[key]) for key in key_list]

            py_func_out = tf.py_func(
                load_fn, inp=[id], Tout=type_out, name='input_py_func')

            return {key: py_func_out[i] for i, key in enumerate(key_list)}

        tf_dataset = tf_dataset.map(load_py_func)

        def set_shape(entry):
            for key in key_list:
                entry[key].set_shape(data_shapes[key])
            return entry
        tf_dataset = tf_dataset.map(set_shape)

    tf_dataset = tf_dataset.padded_batch(
        batch_size, {key: data_shapes[key] for key in key_list})

    tf_dataset = tf_dataset.prefetch(10)

    if is_train:
        tf_dataset = tf_dataset.repeat(1000)  # repeat 1000 epoch

    iterator = tf_dataset.make_one_shot_iterator()
    batch_ops = iterator.get_next()

    return batch_ops
