"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import zipfile
import io
import numpy
import csv
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

WORKING_DIRECTORY = "/Users/partheinstein/projects/tensorflow/digit"
TRAIN_LABELS = WORKING_DIRECTORY + "train.csv"
TRAIN_IMAGES_DIR = WORKING_DIRECTORY + "train/"
TEST_IMAGES_DIR = WORKING_DIRECTORY + "test/"
rows = 28
cols = 28

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_train_images():
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    num_images = 0
    arr = numpy.ndarray(shape=(784, ), dtype=float)
    print("Reading training files...")
    with open(TRAIN_LABELS, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            f = TRAIN_IMAGES_DIR + row[0]
            with io.open(f, 'rb') as bytestream:
                buf = bytestream.read(rows * cols)
                data = numpy.frombuffer(buf, dtype=numpy.uint8)
                # data is of shape (784,1)
                # print(data.shape)
                arr = numpy.concatenate((arr, data), axis=0)
                # print(data.shape)
                # data = data.reshape(num_images, rows, cols, 1)
                # return data
                num_images = num_images + 1

    print("Reading files done. #files read=", num_images)
    # delete the first random 784 elements
    # this is needed because we are concatenating the data from file to arr
    # but when we reshape below, the arr.shape = (7840784,) instead of
    # required (7840000,)
    print("Deleting the first 784 elements...")
    arr = numpy.delete(arr, numpy.arange(784))
    print("After deleting, shape=", arr.shape)
    print("Reshaping...")
    arr = arr.reshape(num_images, rows, cols, 1)
    print("After reshaping, shape", arr.shape)
    return arr

def extract_test_images():
    dir = os.listdir(TEST_IMAGES_DIR)
    num_images = len(dir)
    filenames = []
    arr = numpy.ndarray(shape=(784, ), dtype=float)
    print("Reading test files...")
    for f in dir:
        filenames.append(f)
        with io.open(TEST_IMAGES_DIR + f, 'rb') as bytestream:
            buf = bytestream.read(rows * cols)
            data = numpy.frombuffer(buf, dtype=numpy.uint8)
            arr = numpy.concatenate((arr, data), axis=0)

    print("Reading files done. #files read=", num_images)
    print("Deleting the first 784 elements...")
    arr = numpy.delete(arr, numpy.arange(784))
    print("After deleting, shape=", arr.shape)
    print("Reshaping...")
    arr = arr.reshape(num_images, rows, cols, 1)
    print("After reshaping, shape", arr.shape)
    return arr, filenames


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_train_labels():
    """Extract the labels into a 1D uint8 numpy array [index]."""

    labels = numpy.ndarray(shape=(1, ), dtype=numpy.uint8)
    with open(TRAIN_LABELS, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            label = numpy.array([row[1]], dtype=numpy.uint8)
            # print(label)
            labels = numpy.concatenate((labels, label), axis=0)
    # get rid of the first element (bogus val from initializing ndarray)
    labels = numpy.delete(labels, 0, 0)
    return dense_to_one_hot(labels, 10)

class DataSet(object):
    def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=tf.float32):
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        self._num_examples = images.shape[0]

          # Convert shape from [num examples, rows, columns, depth]
          # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(dtype=tf.float32):
    class DataSets(object):
        pass
    data_sets = DataSets()

    VALIDATION_SIZE = 5000
    train_images = extract_train_images()
    train_labels = extract_train_labels()
    test_images, test_image_filenames = extract_test_images()

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
    data_sets.validation = DataSet(validation_images, validation_labels, dtype=dtype)
    data_sets.test = DataSet(test_images, test_image_filenames, dtype=dtype)

    return data_sets
