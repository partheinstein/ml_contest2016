"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import io
import numpy
import csv
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(folder):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    dir = "/Users/partheinstein/projects/tensorflow/digit"
    print('Looking at ', dir)

    rows = 28
    cols = 28
    num_images = 0
    arr = numpy.ndarray(shape=(784, ), dtype=float)
    print("Reading files...")
    with open(dir + '/' + 'train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            f = dir + '/' + folder + '/' + row[0]
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

    print("Reading files done. files read=", num_images)
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

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_labels(one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    dir = "/Users/partheinstein/projects/tensorflow/digit"

    labels = numpy.ndarray(shape=(1, ), dtype=numpy.uint8)
    with open(dir + '/' + 'train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            label = numpy.array([row[1]], dtype=numpy.uint8)
            # print(label)
            labels = numpy.concatenate((labels, label), axis=0)
    # get rid of the first element (bogus val from initializing ndarray)
    labels = numpy.delete(labels, 0, 0)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels
