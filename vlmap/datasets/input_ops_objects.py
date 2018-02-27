import tensorflow as tf

from util import log


def create(dataset,
           batch_size,
           is_training=False,
           scope='objects_input',
           shuffle=True):

    ids = dataset.ids
    log.info('input_ops {}: using {} IDs from dataset'.format(
        scope, len(ids)))

    with tf.device('/cpu:0'), tf.name_scope(scope):
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(ids))

        def load_fn(id):
            image, sampled_objects, sampled_objects_len, ground_truth, \
                sampled_objects_name = dataset.get_data(id)
            return (id, image, sampled_objects, sampled_objects_len,
                    ground_truth, sampled_objects_name)

        def load_py_func(id):
            py_func_out = tf.py_func(
                load_fn, inp=[id],
                Tout=[tf.string, tf.float32, tf.int32, tf.int32,
                      tf.float32, tf.string],
                name='input_py_func')
            return {
                'id': py_func_out[0],
                'image': py_func_out[1],
                'objects': py_func_out[2],
                'objects_len': py_func_out[3],
                'ground_truth': py_func_out[4],
                'objects_name': py_func_out[5],
            }
        tf_dataset = tf_dataset.map(load_py_func)

        data_shapes = dataset.get_data_shapes()

        def set_shape(entry):
            entry['id'].set_shape([])
            entry['image'].set_shape(data_shapes['image'])
            entry['objects'].set_shape(data_shapes['sampled_objects'])
            entry['objects_len'].set_shape(data_shapes['sampled_objects_len'])
            entry['ground_truth'].set_shape(data_shapes['ground_truth'])
            entry['objects_name'].set_shape(data_shapes['sampled_objects_name'])
            return entry
        tf_dataset = tf_dataset.map(set_shape)

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=10000)
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.repeat(1000)  # repeat 1000 epoch

    iterator = tf_dataset.make_one_shot_iterator()
    batch_ops = iterator.get_next()

    return batch_ops
