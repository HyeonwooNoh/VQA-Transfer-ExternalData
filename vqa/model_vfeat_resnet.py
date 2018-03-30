import tensorflow as tf

from vlmap import modules

ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'


class Model(object):

    def __init__(self, batch, config, is_train=True):
        self.batch = batch
        self.config = config
        self.data_cfg = config.dataset_config

        self.outputs = {}
        self.mid_result = {}
        self.vis_image = {}

        self.build(is_train=is_train)

    def get_enc_I_param_path(self):
        return ENC_I_PARAM_PATH

    def build(self, is_train=False):
        """
        build network architecture and loss
        """

        """
        Visual features
        """
        # feat_V
        enc_I = modules.encode_I_block3(self.batch['image'],
                                        is_train=is_train)
        roi_ft = modules.roi_pool(enc_I, self.batch['normal_box'],
                                  height=1, width=1)
        num_box = tf.shape(self.batch['normal_box'])[1]
        V_ft = tf.reshape(roi_ft, [-1, num_box, enc_I.get_shape().as_list()[3]])

        self.outputs['V_ft'] = V_ft
