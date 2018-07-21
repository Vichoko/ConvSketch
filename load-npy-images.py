import random

import numpy
import matplotlib.pyplot as plt
import os
import pathlib

from time import sleep


def main():
    # elegir 100 clases
    data_path = pathlib.PureWindowsPath('D:/dataset/quickdraw/numpy_bitmap')
    data_dir = os.listdir(data_path)
    chosen_classes_filenames = random.choices(data_dir, k=100)
    chosen_classes_bitmaps = []

    # train: obtener 1000 fotos de cada clase
    for class_filename in chosen_classes_filenames:
        class_bitmaps = numpy.load(str(data_path / class_filename), 'r')
        chosen_classes_bitmaps.append(class_bitmaps[0:1000])

    # plot a sample
    for i, class_bitmaps in enumerate(chosen_classes_bitmaps):
        print(chosen_classes_filenames[i])
        plt.imshow(numpy.reshape(class_bitmaps[0], (28, 28)), cmap='gray')
        plt.show()
        sleep(1)


if __name__ == "__main__":
    main()
