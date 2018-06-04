# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nelson Higuera

"""

import tensorflow as tf

from . import layers


# mnist net
def net(input_shape=[None, 128, 128]):
    # placeholder for is_training = [False or True]. False is used for testing and True for training
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    # placeholder for input data
    x = tf.placeholder(tf.float32, input_shape, name='x')
    # placeholder for labels
    y_true = tf.placeholder(tf.float32, [None, 10], name='y_true')
    # reshape input to fit a  4D tensor
    x_tensor = tf.reshape(x, [-1, x.get_shape().as_list()[1], x.get_shape().as_list()[2], 1])
    # conv_1
    conv1_1 = layers.conv_layer(x_tensor, shape=[3, 3, 1, 64], name='conv1_1', is_training=is_training);
    conv1_2 = layers.conv_layer(conv1_1, shape=[3, 3, 64, 64], name='conv1_2', is_training=is_training);
    maxpool1 = layers.max_pool_layer(conv1_2, 3, 3, stride=2)  # 14 x 14
    print(" conv_1: {} ".format(maxpool1.get_shape().as_list()))
    # conv_2
    conv2_1 = layers.conv_layer(maxpool1, shape=[3, 3, 64, 128], name='conv2_1', is_training=is_training)
    conv2_2 = layers.conv_layer(conv2_1, shape=[3, 3, 128, 128], name='conv2_2', is_training=is_training);
    maxpool2 = layers.max_pool_layer(conv2_2, 3, 3, stride=2)  # 7 x 7
    print(" conv_2: {} ".format(maxpool2.get_shape().as_list()))
    # conv_3
    conv3_1 = layers.conv_layer(maxpool2, shape=[3, 3, 128, 128], name='conv3_1', is_training=is_training)
    conv3_2 = layers.conv_layer(conv3_1, shape=[3, 3, 128, 128], name='conv3_2', is_training=is_training)
    maxpool3 = layers.max_pool_layer(conv3_2, 3, 3, stride=2)  # 3 x 3
    print(" conv_3: {} ".format(maxpool3.get_shape().as_list()))
    # conv_4
    conv4_1 = layers.conv_layer(maxpool3, shape=[3, 3, 128, 256], name='conv4_1', is_training=is_training)
    conv4_2 = layers.conv_layer(conv4_1, shape=[3, 3, 256, 256], name='conv4_2', is_training=is_training)
    maxpool4 = layers.max_pool_layer(conv4_2, 3, 3, stride=2)  # 3 x 3
    print(" conv_4: {} ".format(maxpool4.get_shape().as_list()))
    # fully connected
    fc_1 = layers.fc_layer(maxpool4, 1024, name='fc_1')
    print(" fc_1: {} ".format(fc6.get_shape().as_list()))
    # fully connected
    fc_2 = layers.fc_layer(fc_1, 100, name='fc_2', use_relu=False)
    print(" fc_2: {} ".format(fc_2.get_shape().as_list()))
    # gap = layers.gap_layer(conv_5) # 8x8
    # print(" gap: {} ".format(gap.get_shape().as_list()))
    y_pred = tf.nn.softmax(fc_2)
    # loss function-------------------------
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_2, labels=y_true)
    loss = tf.reduce_mean(cross_entropy)
    # ---------------------------------------
    # accuracy
    y_pred_cls = tf.argmax(y_pred, 1)
    y_true_cls = tf.argmax(y_true, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_true_cls), tf.float32))

    return {'is_training': is_training, 'x': x, 'y_true': y_true, 'y_pred': y_pred, 'loss': loss, 'acc': acc}
