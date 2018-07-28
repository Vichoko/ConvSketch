from configuration_sketch import data_path, IMAGE_DIMENSIONS
from utils.data import create_sketch_tfrecords_from_npy_dumps
import os
import tensorflow as tf


def test_tfrecords():
    counter = 0
    iter = tf.python_io.tf_record_iterator(os.path.join(str(data_path), "train.tfrecords"))
    for example in iter:
        counter += 1
        print("data no: {}".format(counter))
        result = tf.train.Example.FromString(example)
        print(result.features.feature['train/label'].int64_list.value)

        #image = tf.decode_raw(result.features.feature['train/image'].bytes_list.tostring(), tf.uint8)
        image = tf.image.decode_image(result.features.feature['train/image'].bytes_list.value)
        image = tf.reshape(image, [IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]])
        print(image)
if __name__ == "__main__":
    create_sketch_tfrecords_from_npy_dumps()
    #test_tfrecords()
