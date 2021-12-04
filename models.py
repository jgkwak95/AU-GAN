from collections import namedtuple
from utils import *
from ops import *
import time
from glob import glob


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def generator_resnet(image, options, transfer=False, reuse=False, name="generator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block_dilated(x, dim, ks=3, s=1, name='res', down=False):
            if down:
                dim = dim * 2
            y = instance_norm(dilated_conv2d(x, dim, ks, s, padding='SAME', name=name + '_c1'), name + '_bn1')
            y = tf.nn.relu(y)
            y = instance_norm(dilated_conv2d(y, dim, ks, s, padding='SAME', name=name + '_c2'), name + '_bn2')
            out = y + x
            if down:
                out = tf.nn.relu(instance_norm(conv2d(out, dim // 2, 3, 1, name=name + '_down_c'), name + '_in_down'))
            return out

        def residual_block(x_init, dim, ks=3, s=1, name='resblock', down=False):
            with tf.variable_scope(name):
                if down:
                    dim = dim * 2

                with tf.variable_scope('res1'):
                    x = instance_norm(conv2d(x_init, dim, ks, s, padding='SAME', name=name + '_c1'), name + '_in1')
                    x = tf.nn.relu(x)

                with tf.variable_scope('res2'):

                    x = instance_norm(conv2d(x, dim, ks, s, padding='SAME', name=name + '_c2'), name + '_in2')

                out = x + x_init

                if down:
                    out = tf.nn.relu(
                        instance_norm(conv2d(out, dim // 2, 3, 1, name=name + '_down_c'), name + '_in_down'))
                return out

        ### Encoder architecture
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        r1 = residule_block_dilated(c3, options.gf_dim * 4, name='g_r1')
        r2 = residule_block_dilated(r1, options.gf_dim * 4, name='g_r2')
        r3 = residule_block_dilated(r2, options.gf_dim * 4, name='g_r3')
        r4 = residule_block_dilated(r3, options.gf_dim * 4, name='g_r4')
        # r5 = residule_block_dilated(r4, options.gf_dim * 4, name='g_r5')

        if transfer:
            t1 = residual_block(r4, options.gf_dim * 4, name='g_t1')
            t2 = residual_block(t1, options.gf_dim * 4, name='g_t2')
            t3 = residual_block(t2, options.gf_dim * 4, name='g_t3')
            t4 = residual_block(t3, options.gf_dim * 4, name='g_t4')
            # feature = tf.concat([r4, t4], axis=3, name='g_concat')
            # down = True
            feature = t4
        else:
            feature = r4
            t4 = None
            down = False

        ### translation decoder architecture
        r6 = residule_block_dilated(feature, options.gf_dim * 4, name='g_r6')
        r7 = residule_block_dilated(r6, options.gf_dim * 4, name='g_r7')
        r8 = residule_block_dilated(r7, options.gf_dim * 4, name='g_r8')
        r9 = residule_block_dilated(r8, options.gf_dim * 4, name='g_r9')
        d1 = deconv2d(r9, options.gf_dim * 2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        ### reconstruction decoder architecture
        r5 = gaussian_noise_layer(r4, 0.02)
        r6_rec = residule_block_dilated(r5, options.gf_dim * 4, name='g_r6_rec')
        r6_rec = gaussian_noise_layer(r6_rec, 0.02)
        r7_rec = residule_block_dilated(r6_rec, options.gf_dim * 4, name='g_r7_rec')
        r8_rec = residule_block_dilated(r7_rec, options.gf_dim * 4, name='g_r8_rec')
        r9_rec = residule_block_dilated(r8_rec, options.gf_dim * 4, name='g_r9_rec')
        d1_rec = deconv2d(r9_rec, options.gf_dim * 2, 3, 2, name='g_d1_dc_rec')
        d1_rec = tf.nn.relu(instance_norm(d1_rec, 'g_d1_bn_rec'))
        d2_rec = deconv2d(d1_rec, options.gf_dim, 3, 2, name='g_d2_dc_rec')
        d2_rec = tf.nn.relu(instance_norm(d2_rec, 'g_d2_bn_rec'))
        d2_rec = tf.pad(d2_rec, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred_rec = tf.nn.tanh(conv2d(d2_rec, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c_rec'))

        ## confidence prediction

        if transfer:

            d_conf = deconv2d(d1, options.gf_dim, 3, 2, name='g_d_dc_conf')
            d_conf = tf.nn.relu(instance_norm(d_conf, 'g_d_bn_conf'))
            d_conf = tf.pad(d_conf, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            pred_conf = tf.nn.softplus(conv2d(d_conf, 1, 7, 1, padding='VALID', name='g_pred_c_conf'))

        else:
            pred_conf = None

        return pred, pred_rec, r4, t4, pred_conf

def discriminator(image, options, n_scale=2, reuse=False, name="discriminator"):
    images = []
    for i in range(n_scale):
        images.append(
            tf.image.resize_bicubic(image, [get_shape(image)[1] // (2 ** i), get_shape(image)[2] // (2 ** i)]))
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        images = dis_down(images, 4, 2, n_scale, options.df_dim, 'd_h0_conv_scale_')
        images = dis_down(images, 4, 2, n_scale, options.df_dim * 2, 'd_h1_conv_scale_')
        images = dis_down(images, 4, 2, n_scale, options.df_dim * 4, 'd_h2_conv_scale_')
        images = dis_down(images, 4, 2, n_scale, options.df_dim * 8, 'd_h3_conv_scale_')
        images = final_conv(images, n_scale, "d_pred_scale_")
        return images