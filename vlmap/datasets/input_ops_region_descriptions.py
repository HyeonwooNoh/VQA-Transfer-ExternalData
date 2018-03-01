import tensorflow as tf

from util import log


def create(dataset,
           batch_size,
           is_train=False,
           scope='regions_input',
           shuffle=True):

    ids = dataset.ids
    log.info('input_ops {}: using {} IDs from dataset'.format(
        scope, len(ids)))

    with tf.device('/cpu:0'), tf.name_scope(scope):
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(ids))

        def load_fn(id):
            image, region_description, region_description_len = \
                dataset.get_data(id)
            return (id, image, region_description, region_description_len)

        def load_py_func(id):
            py_func_out = tf.py_func(
                load_fn, inp=[id],
                Tout=[tf.string, tf.float32, tf.int32, tf.int32],
                name='input_py_func')
            return {
                'id': py_func_out[0],
                'image': py_func_out[1],
                'region_description': py_func_out[2],
                'region_description_len': py_func_out[3],
            }
        tf_dataset = tf_dataset.map(load_py_func)

        data_shapes = dataset.get_data_shapes()

        def set_shape(entry):
            entry['id'].set_shape([])
            entry['image'].set_shape(data_shapes['image'])
            entry['region_description'].set_shape(
                data_shapes['region_description'])
            entry['region_description_len'].set_shape(
                data_shapes['region_description_len'])
            return entry
        tf_dataset = tf_dataset.map(set_shape)

    if is_train and shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=3000)

    tf_dataset = tf_dataset.batch(batch_size)

    if is_train:
        tf_dataset = tf_dataset.repeat(1000)  # repeat 1000 epoch

    iterator = tf_dataset.make_one_shot_iterator()
    batch_ops = iterator.get_next()

    return batch_ops
