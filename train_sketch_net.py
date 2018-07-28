# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:30:08 2018
@author: jose.saavedra

A convolutional neural network for mnist
This uses Dataset and Estimator components from tensorflow

"""
import argparse
import os

import numpy as np
import tensorflow as tf

import utils.data as data
import utils.sketch_model as mnistnet
# define the input function
from configuration_sketch import data_path, SNAPSHOT_PREFIX, IMAGE_DIMENSIONS, CLASSES_COUNT, SNAPSHOT_TIME, \
    LEARNING_RATE, ESTIMATED_NUMBER_OF_BATCHES, BATCH_SIZE, NUM_ITERATIONS, TEST_TIME


def input_fn(filename, image_shape, mean_img, is_training):
    def parser_tfrecord_sk(serialized_example, mean_img):
        features = tf.parse_example([serialized_example],
                                    features={
                                        'train/image': tf.FixedLenFeature([], tf.string),
                                        'train/label': tf.FixedLenFeature([], tf.int64),
                                    })
        image = tf.decode_raw(features['train/image'], tf.uint8)
        image = tf.reshape(image, [IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]])
        image = tf.cast(image, tf.float32) - tf.cast(tf.constant(mean_img), tf.float32)
        # image = image * 1.0 / 255.0
        # one-hot
        label = tf.one_hot(tf.cast(features['train/label'], tf.int32), CLASSES_COUNT)
        label = tf.reshape(label, [CLASSES_COUNT])
        label = tf.cast(label, tf.float32)
        return image, label

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(
        lambda x: parser_tfrecord_sk(x, mean_img))
    dataset = dataset.batch(BATCH_SIZE)
    if is_training:
        dataset = dataset.shuffle(ESTIMATED_NUMBER_OF_BATCHES)
        epochs_no = int(NUM_ITERATIONS * 1.0 / ESTIMATED_NUMBER_OF_BATCHES)
        dataset = dataset.repeat(epochs_no)
        # for testing shuffle and repeat are not required
    return dataset


# -----------main----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training / testing xk models")
    parser.add_argument("-mode", type=str, choices=['test', 'train'], help=" test | train ", required=True)
    parser.add_argument("-device", type=str, choices=['cpu', 'gpu'], help=" cpu | gpu ", required=True)
    # parser.add_argument("-arch", type=str, help=" name of section in the configuration file", required=True)
    parser.add_argument("-ckpt", type=str,
                        help=" <optional>, it defines the checkpoint for training<fine tuning> or testing",
                        required=False)
    pargs = parser.parse_args()
    run_mode = pargs.mode
    device_name = "/" + pargs.device + ":0"
    # verifying if output path exists
    if not os.path.exists(os.path.dirname(SNAPSHOT_PREFIX)):
        os.makedirs(os.path.dirname(SNAPSHOT_PREFIX))
        # metadata
    filename_mean = os.path.join(str(data_path), "mean.dat")
    # reading metadata
    image_shape = IMAGE_DIMENSIONS
    number_of_classes = CLASSES_COUNT

    # load mean
    mean_img = np.fromfile(filename_mean, dtype=np.float64)
    mean_img = np.reshape(mean_img, image_shape.tolist())
    # defining files for training and test
    filename_train = os.path.join(str(data_path), "train.tfrecords")
    filename_test = os.path.join(str(data_path), "train.tfrecords")

    # -using device gpu or cpu
    with tf.device(device_name):
        estimator_config = tf.estimator.RunConfig(model_dir=SNAPSHOT_PREFIX,
                                                  save_checkpoints_steps=SNAPSHOT_TIME,
                                                  keep_checkpoint_max=10)

        classifier = tf.estimator.Estimator(model_fn=mnistnet.model_fn,
                                            config=estimator_config,
                                            params={'learning_rate': LEARNING_RATE,
                                                    'number_of_classes': CLASSES_COUNT,
                                                    'image_shape': image_shape,
                                                    'model_dir': SNAPSHOT_PREFIX,
                                                    'ckpt': pargs.ckpt
                                                    }
                                            )
        #
        tf.logging.set_verbosity(tf.logging.INFO)  # Just to have some logs to display for demonstration
        # training
        if run_mode == 'train':
            train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(filename_train,
                                                                          image_shape,
                                                                          mean_img,
                                                                          is_training=True),
                                                max_steps=NUM_ITERATIONS)
            # max_steps is not usefule when inherited checkpoint is used
            eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(filename_test,
                                                                        image_shape,
                                                                        mean_img,
                                                                        is_training=False),
                                              start_delay_secs=TEST_TIME,
                                              throttle_secs=TEST_TIME * 2)
            #
            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

        # testing
        if run_mode == 'test':
            result = classifier.evaluate(
                input_fn=lambda: input_fn(filename_test, image_shape, mean_img, is_training=False),
                checkpoint_path=pargs.ckpt)
            print(result)

    print("ok")
