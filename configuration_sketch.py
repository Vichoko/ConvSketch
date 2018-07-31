"""
Configuration file
"""
# Sketch
import pathlib

import numpy
net_type = "resnet" # options: "resnet" and "net


NUM_ITERATIONS = 4000  # 1 iteration =  1 batch
BATCH_SIZE = 10
ESTIMATED_NUMBER_OF_BATCHES = 600
ESTIMATED_NUMBER_OF_BATCHES_TEST = 100
SNAPSHOT_TIME = 1000
TEST_TIME = 60
SNAPSHOT_PREFIX = './snapshot_trained_sketch'

data_path = pathlib.PureWindowsPath('D:/dataset/quickdraw/ndjson')
CLASSES_COUNT = 100
TRAIN_EXAMPLES_PER_CLASS = 1000
TEST_EXAMPLES_PER_CLASS = 50

IMAGE_DIMENSIONS = numpy.array([128, 128])

LEARNING_RATE = 0.0001
DRAW_IMAGE_LIMIT = 10000

