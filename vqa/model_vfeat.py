import tensorflow as tf

from vlmap import modules

V_DIM = 512

ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'

ROI_SZ = 5


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
        I_lowdim = modules.I_reduce_dim(enc_I, V_DIM, scope='I_reduce_dim',
                                        is_train=is_train)
        roi_ft = modules.roi_pool(I_lowdim, self.batch['normal_box'],
                                  ROI_SZ, ROI_SZ)

        roi_ft_flat = tf.reshape(roi_ft, [-1, ROI_SZ, ROI_SZ, V_DIM])

        V_ft_flat = modules.I2V(roi_ft_flat, V_DIM, V_DIM, is_train=is_train)

        num_box = tf.shape(self.batch['normal_box'])[1]
        V_ft = tf.reshape(V_ft_flat, [-1, num_box, V_DIM])

        self.outputs['V_ft'] = V_ft
