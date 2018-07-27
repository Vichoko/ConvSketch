"""
Configuration file
"""
# Sketch
import pathlib

NUM_ITERATIONS = 1000 # 1 iteration =  1 batch
DATASET_SIZE = 60000  # for training
BATCH_SIZE = 100
ESTIMATED_NUMBER_OF_BATCHES = 600
ESTIMATED_NUMBER_OF_BATCHES_TEST = 100
SNAPSHOT_TIME = 1000
TEST_TIME = 600
SNAPSHOT_PREFIX = './snapshot_trained_sketch'
DATA_DIR = 'D:/dataset/quickdraw/numpy_bitmap'
data_path = pathlib.PureWindowsPath('D:/dataset/quickdraw/numpy_bitmap')

