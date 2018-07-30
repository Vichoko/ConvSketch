"""
Configuration file
"""
# Sketch
import pathlib

import numpy

NUM_ITERATIONS = 10000 # 1 iteration =  1 batch
BATCH_SIZE = 100
ESTIMATED_NUMBER_OF_BATCHES = 600
ESTIMATED_NUMBER_OF_BATCHES_TEST = 100
SNAPSHOT_TIME = 1000
TEST_TIME = 60
SNAPSHOT_PREFIX = './snapshot_trained_sketch'
DATA_DIR = 'D:/dataset/quickdraw/numpy_bitmap'

data_path = pathlib.PureWindowsPath('D:/dataset/quickdraw/numpy_bitmap')
CLASSES_COUNT = 100
TRAIN_EXAMPLES_PER_CLASS = 1000
TEST_EXAMPLES_PER_CLASS = 50

IMAGE_DIMENSIONS = numpy.array([28, 28])

LEARNING_RATE = 0.0001


