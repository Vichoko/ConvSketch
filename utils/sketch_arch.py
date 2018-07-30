#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:38:03 2018

@author: Jose M. Saavedra
"""

import tensorflow as tf
from . import layers


# A net for mnist classification
# features: containing feature vectors to be trained
# input_shape: [height, width]
# n_classes int
# is_training: boolean [it should be True for training and False for testing]
def sknet_fn(features, input_shape, n_classes, is_training=True):
    with tf.variable_scope("model_scope"):
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1], 1])
        # conv_1
        conv1_1 = layers.conv_layer(x_tensor, shape=[3, 3, 1, 64], name='conv1_1', is_training=is_training);
        conv1_2 = layers.conv_layer(conv1_1, shape=[3, 3, 64, 64], name='conv1_2', is_training=is_training);
        maxpool1 = layers.max_pool_layer(conv1_2, 3, 2)  # 14 x 14
        print(" conv_1: {} ".format(maxpool1.get_shape().as_list()))
        # conv_2
        conv2_1 = layers.conv_layer(maxpool1, shape=[3, 3, 64, 128], name='conv2_1', is_training=is_training)
        conv2_2 = layers.conv_layer(conv2_1, shape=[3, 3, 128, 128], name='conv2_2', is_training=is_training);
        maxpool2 = layers.max_pool_layer(conv2_2, 3, 2)  # 7 x 7
        print(" conv_2: {} ".format(maxpool2.get_shape().as_list()))
        # conv_3
        conv3_1 = layers.conv_layer(maxpool2, shape=[3, 3, 128, 128], name='conv3_1', is_training=is_training)
        conv3_2 = layers.conv_layer(conv3_1, shape=[3, 3, 128, 128], name='conv3_2', is_training=is_training)
        maxpool3 = layers.max_pool_layer(conv3_2, 3, 2)  # 3 x 3
        print(" conv_3: {} ".format(maxpool3.get_shape().as_list()))
        # conv_4
        conv4_1 = layers.conv_layer(maxpool3, shape=[3, 3, 128, 256], name='conv4_1', is_training=is_training)
        conv4_2 = layers.conv_layer(conv4_1, shape=[3, 3, 256, 256], name='conv4_2', is_training=is_training)
        maxpool4 = layers.max_pool_layer(conv4_2, 3, 2)  # 3 x 3
        print(" conv_4: {} ".format(maxpool4.get_shape().as_list()))
        # fully connected
        fc_1 = layers.fc_layer(maxpool4, 1024, name='fc_1')
        print(" fc_1: {} ".format(fc_1.get_shape().as_list()))
        # fully connected
        fc_2 = layers.fc_layer(fc_1, n_classes, name='fc_2', use_relu=False)
        print(" fc_2: {} ".format(fc_2.get_shape().as_list()))

    return {"output": fc_2, "deep_feature": fc_1}


def skresnet_fn(features, input_shape, n_classes, is_training=True):
    with tf.variable_scope("model_scope"):
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1], 1])
        # conv_1
        conv1_1 = layers.conv_layer(x_tensor, shape=[3, 3, 1, 64], name='conv1_1', is_training=is_training);
        conv1_2 = layers.conv_layer(conv1_1, shape=[3, 3, 64, 64], name='conv1_2', is_training=is_training);
        maxpool1 = layers.max_pool_layer(conv1_2, 3, 2)  # 14 x 14
        print(" conv_1: {} ".format(maxpool1.get_shape().as_list()))

        # conv_2
        conv2_1 = layers.conv_layer(maxpool1, shape=[3, 3, 64, 64], name='conv2_1', is_training=is_training)
        conv2_2 = layers.conv_layer(conv2_1, shape=[3, 3, 64, 64], name='conv2_2', is_training=is_training);
        residual_1 = conv2_2 + maxpool1  # 14 x 14
        print(" conv_2: {} ".format(residual_1.get_shape().as_list()))

        # conv_3
        conv3_1 = layers.conv_layer(residual_1, shape=[3, 3, 64, 64], name='conv3_1', is_training=is_training)
        conv3_2 = layers.conv_layer(conv3_1, shape=[3, 3, 64, 64], name='conv3_2', is_training=is_training)
        residual_2 = conv3_2 + residual_1  # 14 x 14
        print(" conv_3: {} ".format(residual_2.get_shape().as_list()))

        # conv_4
        conv4_1 = layers.conv_layer(residual_2, shape=[3, 3, 64, 128], name='conv4_1', is_training=is_training)
        maxpool2 = layers.max_pool_layer(conv4_1, 3, 2)  # 7 x 7
        print(" conv_4: {} ".format(maxpool2.get_shape().as_list()))

        # conv_5
        conv5_1 = layers.conv_layer(maxpool2, shape=[3, 3, 128, 128], name='conv2_1', is_training=is_training)
        conv5_2 = layers.conv_layer(conv5_1, shape=[3, 3, 128, 128], name='conv2_2', is_training=is_training);
        residual_3 = conv5_2 + maxpool2  # 7 x 7
        print(" conv_5: {} ".format(residual_3.get_shape().as_list()))

        # conv_6
        conv6_1 = layers.conv_layer(residual_3, shape=[3, 3, 128, 128], name='conv3_1', is_training=is_training)
        conv6_2 = layers.conv_layer(conv6_1, shape=[3, 3, 128, 128], name='conv3_2', is_training=is_training)
        residual_4 = conv6_2 + residual_3  # 7 x 7
        print(" conv_6: {} ".format(residual_4.get_shape().as_list()))

        # conv_7
        conv7_1 = layers.conv_layer(residual_4, shape=[3, 3, 128, 256], name='conv4_1', is_training=is_training)
        maxpool3 = layers.max_pool_layer(conv7_1, 3, 2)  # 3 x 3
        print(" conv_7: {} ".format(maxpool3.get_shape().as_list()))

        # conv_8
        conv8_1 = layers.conv_layer(maxpool3, shape=[3, 3, 256, 256], name='conv2_1', is_training=is_training)
        conv8_2 = layers.conv_layer(conv8_1, shape=[3, 3, 256, 256], name='conv2_2', is_training=is_training);
        residual_5 = conv8_2 + maxpool3  # 3 x 3
        print(" conv_8: {} ".format(residual_5.get_shape().as_list()))

        # conv_9
        conv9_1 = layers.conv_layer(residual_5, shape=[3, 3, 256, 256], name='conv3_1', is_training=is_training)
        conv9_2 = layers.conv_layer(conv9_1, shape=[3, 3, 256, 256], name='conv3_2', is_training=is_training)
        residual_6 = conv9_2 + residual_5  # 3 x 3
        print(" conv_9: {} ".format(residual_6.get_shape().as_list()))

        # conv_10
        conv10_1 = layers.conv_layer(residual_6, shape=[3, 3, 256, 256], name='conv4_1', is_training=is_training)
        maxpool4 = layers.max_pool_layer(conv10_1, 3, 2)  # 3 x 3
        print(" conv_10: {} ".format(maxpool4.get_shape().as_list()))

        # fully connected
        fc_1 = layers.fc_layer(maxpool4, 1024, name='fc_1')
        print(" fc_1: {} ".format(fc_1.get_shape().as_list()))
        # fully connected
        fc_2 = layers.fc_layer(fc_1, n_classes, name='fc_2', use_relu=False)
        print(" fc_2: {} ".format(fc_2.get_shape().as_list()))

    return {"output": fc_2, "deep_feature": fc_1}