# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import trend_data
import tensorflow as tf
import csv

mnist = trend_data.read_data_sets()

# create a session
sess = tf.InteractiveSession()

# Create the model
# ----------------

# here None means that dimension will be filled later when we input the images
# x is the input
x = tf.placeholder(tf.float32, [None, 784])

# weights - every pixel is given a weight for a digit.
W = tf.Variable(tf.zeros([784, 10]))

# bias
b = tf.Variable(tf.zeros([10]))

# softmax is the activation func
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
# -------------------------
# y_ is the truth, it is a one-hot vector where [0,0,1,...] means digit 2
# also called the one hot vector
y_ = tf.placeholder(tf.float32, [None, 10])

# the cost function is (y_ * tf.log(y)) and we want to reduce it
# reduce_sum computes the sum of elements across dimensions
# reduce_sum(cost_func, 1) = [sum of elements in row1, sum of elements in row2, ...] etc
# reduce_mean computes the mean
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# the gradient descent works by going in the direction that reduces the cost function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train
# -------
tf.initialize_all_variables().run()

# stochastic training
# take random batches of 100 from the training set
for i in range(1000):
  # here xs are the images and the ys are the labels
  # next_batch is defined in mnist.py
  batch_xs, batch_ys = mnist.train.next_batch(100)

  # train_step is declared above
  # it is a func that takes input x, one hot vector y
  train_step.run({x: batch_xs, y_: batch_ys})

# Test
# ----
actual_probabilities = y.eval({x: mnist.test.images})

sess.close()

with open('test.csv', 'wb') as csvfile:
    i = 0
    csvwriter = csv.writer(csvfile, delimiter=',')
    for filename in mnist.test.labels:
        csvwriter.writerow([filename, actual_probabilities[i]])
        i = i + 1
