import h5py
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from util import log

GLOVE_EMBEDDING_PATH = 'data/preprocessed/glove.6B.300d.hdf5'
ENC_I_R_MEAN = 123.68
ENC_I_G_MEAN = 116.78
ENC_I_B_MEAN = 103.94


def language_encoder(seq, seq_len, dim=384, scope='language_encoder', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if not reuse: log.warning(scope.name)
        cell = rnn.BasicLSTMCell(num_units=dim, state_is_tuple=True)
        _, final_state = tf.nn.dynamic_rnn(
            cell=cell, dtype=tf.float32, sequence_length=seq_len,
            inputs=seq)
        return final_state.h


def encode_I(images, is_train=False, reuse=False):
    """
    Pre-trained model parameter is available here:
    https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
    """
    with tf.name_scope('enc_I_preprocess'):
        channels = tf.split(axis=3, num_or_size_splits=3, value=images)
        for i, mean in enumerate([ENC_I_R_MEAN, ENC_I_G_MEAN, ENC_I_B_MEAN]):
            channels[i] -= mean
        processed_I = tf.concat(axis=3, values=channels)

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        enc_I, _ = nets.resnet_v1.resnet_v1_50(
            processed_I,
            is_training=is_train,
            global_pool=True,
            output_stride=None,
            reuse=reuse,
            scope='resnet_v1_50')
        enc_I = tf.squeeze(enc_I, axis=[1, 2])
    return enc_I


def I2V(enc_I, enc_dim, out_dim, scope='I2V', is_train=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if not reuse: log.warning(scope.name)
        feat_V = fc_layer(
            enc_I, enc_dim, use_bias=False, use_bn=True,
            activation_fn=tf.nn.relu, is_training=is_train,
            scope='fc_1', reuse=False)
        feat_V = fc_layer(
            feat_V, out_dim, use_bias=True, use_bn=False,
            activation_fn=None, is_training=is_train,
            scope='Linear', reuse=False)
        return feat_V


def word_prediction(inputs, word_weights, activity_regularizer=None,
                    trainable=True, name=None, reuse=None):
    layer = WordPredictor(word_weights, activity_regularizer=activity_regularizer,
                          trainable=trainable, name=name,
                          dtype=inputs.dtype.base_dtype,
                          _scope=name, _reuse=reuse)
    return layer.apply(inputs)

def batch_word_classifier(inputs, word_weights, scope='batch_word_classifier',
                          reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if not reuse: log.warning(scope.name)
        bias = tf.get_variable(name='bias', shape=(),
                               initializer=tf.zeros_initializer())
        logits = tf.reduce_sum(tf.expand_dims(inputs, axis=1) * word_weights,
                               axis=-1) + bias
        return logits


class WordPredictor(tf.layers.Layer):

    def __init__(self,
                 word_weights,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(WordPredictor, self).__init__(
            trainable=trainable, name=name,
            activity_regularizer=activity_regularizer, **kwargs)

        self.word_weights = word_weights  # [num_words, dim]
        self.num_words = word_weights.get_shape().as_list()[0]
        if self.num_words is None:
            raise ValueError(
                'The first dimension of the weights must be defined')

    def build(self, input_shape):
        self.bias = self.add_variable('bias',
                                      shape=(),
                                      dtype=self.dtype,
                                      initializer=tf.zeros_initializer(),
                                      regularizer=None,
                                      trainable=True)
        self.built = True

    def call(self, inputs):
        # Inputs: [bs, dim]
        logits = tf.reduce_sum(tf.expand_dim(inputs, axis=1) *
                               tf.expand_dim(self.word_weights, axis=0),
                               axis=-1) + self.bias
        return logits  # [bs, num_words]

    def _compute_output_shape(self, input_shape):
        # [bs, dim]
        input_shape = tf.TensorShape(input_shape).as_list()

        # [bs, num_words]
        output_shape = tf.TensorShape([input_shape[0], self.num_words])
        return output_shape


def language_decoder(init_state, seq, seq_len, embedding_lookup,
                     dim, start_token, end_token, max_seq_len,
                     unroll_type='teacher_forcing',
                     output_layer=None,
                     scope='language_decoder', reuse=False):
    """
    Args:
        init_state: initial cell state
        seq: sequence of token (usually ground truth sequence)
        start_token: integer for start token <s>
        end_token: integer for end token <e>
    """
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if not reuse: log.warning(scope.name)
        if unroll_type == 'teacher_forcing':
            # Teacher forcing helper
            embed = embedding_lookup(seq)
            helper = seq2seq.TrainingHelper(embed, seq_len)
        elif unroll_type == 'greedy':
            start_tokens = tf.zeros([tf.shape(seq)[0]]) + start_token
            helper = seq2seq.GreedyEmbeddingHelper(
                embedding_lookup, start_tokens, end_token)
        else:
            raise ValueError('Unknown unroll_type')

        cell = rnn.BasicLSTMCell(num_units=dim, state_is_tuple=True)
        decoder = seq2seq.BasicDecoder(cell, helper, init_state,
                                       output_layer=output_layer)
        outputs, _, pred_length = seq2seq.dynamic_decode(
            decoder, maximum_iterations=max_seq_len,
            scope='dynamic_decoder')
        # Dynamically pad outputs
        output = outputs.rnn_output
        sz = tf.shape(output)
        dynamic_pad = tf.zeros(
            [sz[0], max_seq_len - sz[1], sz[2]], dtype=output.dtype)
        output = tf.concat([output, dynamic_pad], axis=1)
        sz = output.get_shape().as_list()
        output.set_shape([sz[0], max_seq_len, sz[2]])  # [bs, len, n]

        pred = outputs.sample_id
        sz = tf.shape(pred)
        dynamic_pad = tf.zeros([sz[0], max_seq_len - sz[1]], dtype=pred.dtype)
        pred = tf.concat([pred, dynamic_pad], axis=1)
        sz = pred.get_shape().as_list()
        pred.set_shape([sz[0], max_seq_len])  # [bs, len]

        return output, pred, pred_length


def glove_embedding(seq, scope='glove_embedding', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if not reuse: log.warning(scope.name)
        with h5py.File(GLOVE_EMBEDDING_PATH, 'r') as f:
            fixed = tf.constant(f['param'].value.transpose())
        learn = tf.get_variable(
            name='learn', shape=[3, 300],
            initializer=tf.random_uniform_initializer(
                minval=-0.01, maxval=0.01))
        embed_map = tf.concat([fixed, learn], axis=0)
        embed = tf.nn.embedding_lookup(embed_map, seq)
        return embed


def used_wordset(used_wordset_path, scope='used_wordset'):
    with tf.name_scope(scope):
        log.warning(scope)
        with h5py.File(used_wordset_path, 'r') as f:
            wordset = tf.constant(f['used_wordset'].value, dtype=tf.int32)
        return wordset


def L2V(feat_L, enc_dim, out_dim, is_train=True, scope='L2V', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if not reuse: log.warning(scope.name)
        h = fc_layer(
            feat_L, enc_dim, use_bias=True, use_bn=False,
            activation_fn=tf.nn.relu, is_training=is_train,
            scope='fc_1', reuse=False)
        h = fc_layer(
            h, enc_dim, use_bias=True, use_bn=False,
            activation_fn=tf.nn.relu, is_training=is_train,
            scope='fc_2', reuse=False)
        map_V = fc_layer(
            h, out_dim, use_bias=True, use_bn=False,
            activation_fn=None, is_training=is_train,
            scope='Linear', reuse=False)
        return map_V


def fc_layer(input, dim, use_bias=False, use_bn=False, activation_fn=None,
             is_training=True, scope='fc_layer', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if not reuse: log.warning(scope.name)
        if use_bias:
            out = layers.fully_connected(
                input, dim, activation_fn=None, reuse=reuse,
                trainable=is_training, scope='fc')
        else:
            out = layers.fully_connected(
                input, dim, activation_fn=None, biases_initializer=None,
                reuse=reuse, trainable=is_training, scope='fc')
        if use_bn:
            out = layers.batch_norm(out, center=True, scale=True, decay=0.9,
                                    is_training=is_training,
                                    updates_collections=None)
        if activation_fn is not None:
            out = activation_fn(out)
        return out
