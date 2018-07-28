#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2018
@author: jsaavedr

Description: A list of function to create tfrecords
"""

import os
import random
import struct
import sys

import numpy
import numpy as np
import tensorflow as tf


# %%
from configuration_sketch import data_path, CLASSES_COUNT, TEST_EXAMPLES_PER_CLASS, TRAIN_EXAMPLES_PER_CLASS, \
    IMAGE_DIMENSIONS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# %%
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# %%
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# creating tfrecords
def createTFRecordFromList(images, labels, tfr_filename):
    h = IMAGE_DIMENSIONS[0]
    w = IMAGE_DIMENSIONS[1]
    writer = tf.python_io.TFRecordWriter(tfr_filename)
    assert len(images) == len(labels)
    mean_image = np.zeros([h,w], dtype=np.float32)
    for i in range(len(images)):
        if i % 500 == 0:
            print("---{}".format(i))
        image = images[i].astype(np.uint8)
        img_raw = image.tostring()

        # image = processFun(image, (w, h))
        #create a feature
        feature = {'train/label': _int64_feature(labels[i]),
                   'train/image': _bytes_feature(img_raw)}
        #crate an example protocol buffer
        example = tf.train.Example(features = tf.train.Features(feature=feature))
        #serialize to string an write on the file
        writer.write(example.SerializeToString())
        mean_image = mean_image + image / len(images)
    #serialize mean_image
    writer.close()
    sys.stdout.flush()
    return mean_image





def create_sketch_tfrecords_from_npy_dumps():
    # Config


    # choose 100 classes
    data_dir_ls = os.listdir(data_path)
    data_dir_ls = [element for element in data_dir_ls if ".npy" in element] # list only ".npy" files
    chosen_classes_filenames = random.choices(data_dir_ls, k=CLASSES_COUNT)

    # create accumulators
    train_bitmaps = None
    train_labels = []

    test_bitmaps = None
    test_labels = []

    for idx, class_filename in enumerate(chosen_classes_filenames):
        # load sketches to numpy arrays (accumulators)
        print("processing class {} ({}/{})".format(chosen_classes_filenames[idx].replace(".npy", ""), idx+1, CLASSES_COUNT))
        # load & reshape
        class_bitmaps = numpy.load(str(data_path / class_filename), 'r')
        class_bitmaps = numpy.reshape(class_bitmaps, (class_bitmaps.shape[0], IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]))

        # instance numpy array accumulators
        if train_bitmaps is None:
            train_bitmaps = numpy.empty((0, class_bitmaps.shape[1], class_bitmaps.shape[2]))
        if test_bitmaps is None:
            test_bitmaps = numpy.empty((0, class_bitmaps.shape[1], class_bitmaps.shape[2]))

        # sample bitmaps for train and testing
        train_bitmaps = numpy.concatenate((random.choices(class_bitmaps, k=TRAIN_EXAMPLES_PER_CLASS), train_bitmaps))
        train_labels += [idx for _ in range(TRAIN_EXAMPLES_PER_CLASS)]

        test_bitmaps = numpy.concatenate((random.choices(class_bitmaps, k=TEST_EXAMPLES_PER_CLASS), test_bitmaps))
        test_labels += [idx for _ in range(TEST_EXAMPLES_PER_CLASS)]

    from sklearn.utils import shuffle
    train_bitmaps, train_labels = shuffle(train_bitmaps, train_labels)

    # train and test tf records
    training_mean = createTFRecordFromList(
        train_bitmaps,
        train_labels,
        os.path.join(str(data_path), "train.tfrecords")
    )
    print('info: train.tfrecords saved')
    createTFRecordFromList(
        test_bitmaps,
        test_labels,
        os.path.join(str(data_path), "test.tfrecords")
    )
    print('info: test.tfrecords saved')

    # save mean in file
    mean_file = os.path.join(str(data_path), "mean.dat")
    print("info: mean_file {}".format(training_mean.shape))
    training_mean.tofile(mean_file)
    print("info: mean_file saved at {}.".format(mean_file))