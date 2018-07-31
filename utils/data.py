#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2018
@author: jsaavedr

Description: A list of function to create tfrecords
"""

import os
import random
import sys

import numpy
import numpy as np
import tensorflow as tf
from skimage.draw import line_aa
from skimage.transform import resize
from sklearn.utils import shuffle

import json
# %%
from configuration_sketch import data_path, CLASSES_COUNT, TEST_EXAMPLES_PER_CLASS, TRAIN_EXAMPLES_PER_CLASS, \
    IMAGE_DIMENSIONS, DRAW_IMAGE_LIMIT


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
    mean_image = np.zeros([h, w], dtype=np.float32)
    for i in range(len(images)):
        if i % 500 == 0:
            print("---{}".format(i))
        image = images[i].astype(np.uint8)
        img_raw = image.tostring()

        # image = processFun(image, (w, h))
        # create a feature
        feature = {'train/label': _int64_feature(labels[i]),
                   'train/image': _bytes_feature(img_raw)}
        # crate an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # serialize to string an write on the file
        writer.write(example.SerializeToString())
        mean_image = mean_image + image / len(images)
    # serialize mean_image
    writer.close()
    sys.stdout.flush()
    return mean_image


def createImage(points):
    x_points = []
    y_points = []
    target_size = 256
    object_size = 200
    # reading all points
    for stroke in points:
        x_points = x_points + stroke[0]
        y_points = y_points + stroke[1]
        # min max for each axis
    min_x = min(x_points)
    max_x = max(x_points)
    min_y = min(y_points)
    max_y = max(y_points)

    im_width = np.int(max_x - min_x + 1)
    im_height = np.int(max_y - min_y + 1)

    if im_width > im_height:
        resize_factor = np.true_divide(object_size, im_width)
    else:
        resize_factor = np.true_divide(object_size, im_height)

    t_width = np.int(im_width * resize_factor)
    t_height = np.int(im_height * resize_factor)

    center_x = np.int(sum(x_points) / len(x_points))
    center_y = np.int(sum(y_points) / len(y_points))

    center_x = np.int(t_width * 0.5)
    center_y = np.int(t_height * 0.5)

    t_center_x = np.int(target_size * 0.5)
    t_center_y = np.int(target_size * 0.5)

    offset_x = t_center_x - center_x
    offset_y = t_center_y - center_y

    blank_image = np.zeros((target_size, target_size), np.uint8)
    blank_image[:, :] = 0
    # cv2.circle(blank_image, (), 1, 1, 8)
    for stroke in points:
        xa = -1
        ya = -1
        for p in zip(stroke[0], stroke[1]):
            x = np.int(np.true_divide(p[0] - min_x, im_width) * t_width) + offset_x
            y = np.int(np.true_divide(p[1] - min_y, im_height) * t_height) + offset_y
            # if x in range(0,1024) and y in range(0,1024):
            if xa >= 0 and ya >= 0:
                rr, cc, val = line_aa(xa, ya, x, y)
                blank_image[rr, cc] = val * 255
            xa = x
            ya = y
            blank_image[y, x] = 0
    return blank_image


def create_sketch_tfrecords_from_ndjson():
    # choose 100 classes
    data_dir_ls = os.listdir(data_path)
    data_dir_ls = [element for element in data_dir_ls if ".ndjson" in element]  # list only ".npy" files
    chosen_classes_filenames = random.choices(data_dir_ls, k=CLASSES_COUNT)

    # create accumulators
    train_bitmaps = None
    train_labels = []

    test_bitmaps = None
    test_labels = []

    for idx, class_filename in enumerate(chosen_classes_filenames):
        # load sketches to numpy arrays (accumulators)
        print("processing class {} ({}/{})".format(chosen_classes_filenames[idx].replace(".ndjson", ""), idx + 1,
                                                   CLASSES_COUNT))
        class_bitmaps = []
        with (open(str(data_path / class_filename))) as f:
            for data_counter, str_line in enumerate(f):
                if data_counter >= TRAIN_EXAMPLES_PER_CLASS + TEST_EXAMPLES_PER_CLASS:
                    print('info: DRAW_IMAGE_LIMIT reached')
                    break
                data = json.loads(str_line)
                coords = data["drawing"]
                image = createImage(coords)
                # resize images to 128x128 and cast to int
                image = resize(image, (IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]), preserve_range=True).astype(
                        numpy.uint8)
                class_bitmaps.append(image)

        class_bitmaps = numpy.array(class_bitmaps)
        class_bitmaps = shuffle(class_bitmaps)
        # instance numpy array accumulators
        local_train_bitmaps = class_bitmaps[:TRAIN_EXAMPLES_PER_CLASS]
        local_test_bitmaps = class_bitmaps[TRAIN_EXAMPLES_PER_CLASS:]

        if train_bitmaps is None:
            train_bitmaps = local_train_bitmaps
        else:
            train_bitmaps = numpy.concatenate((local_train_bitmaps, train_bitmaps))

        if test_bitmaps is None:
            test_bitmaps = local_test_bitmaps
        else:
            test_bitmaps = numpy.concatenate((local_test_bitmaps, test_bitmaps))

        # sample bitmaps for train and testing
        train_labels += [idx for _ in range(TRAIN_EXAMPLES_PER_CLASS)]
        test_labels += [idx for _ in range(TEST_EXAMPLES_PER_CLASS)]

    pack_tfrecords(test_bitmaps, test_labels, train_bitmaps, train_labels)


def create_sketch_tfrecords_from_npy_dumps():
    # Config

    # choose 100 classes
    data_dir_ls = os.listdir(data_path)
    data_dir_ls = [element for element in data_dir_ls if ".npy" in element]  # list only ".npy" files
    chosen_classes_filenames = random.choices(data_dir_ls, k=CLASSES_COUNT)

    # create accumulators
    train_bitmaps = None
    train_labels = []

    test_bitmaps = None
    test_labels = []

    for idx, class_filename in enumerate(chosen_classes_filenames):
        # load sketches to numpy arrays (accumulators)
        print("processing class {} ({}/{})".format(chosen_classes_filenames[idx].replace(".npy", ""), idx + 1,
                                                   CLASSES_COUNT))
        # load & reshape to original shape
        class_bitmaps = numpy.load(str(data_path / class_filename), 'r')
        class_bitmaps = numpy.reshape(class_bitmaps, (class_bitmaps.shape[0], 28, 28))

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

    # resize images to 128x128 and cast to int
    aux = []
    for bitmap in train_bitmaps:
        aux.append(resize(bitmap, (IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]), preserve_range=True).astype(
            numpy.uint8))
    train_bitmaps = aux
    aux = []
    for bitmap in test_bitmaps:
        aux.append(resize(bitmap, (IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]), preserve_range=True).astype(
            numpy.uint8))
    test_bitmaps = aux
    pack_tfrecords(test_bitmaps, test_labels, train_bitmaps, train_labels)


def pack_tfrecords(test_bitmaps, test_labels, train_bitmaps, train_labels):
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

