'''
	These are the different models we will use when testing our code. 
	----------------------------------------------------------------
	LICENSE
	#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
	#
	#  Licensed under the Apache License, Version 2.0 (the "License");
	#  you may not use this file except in compliance with the License.
	#  You may obtain a copy of the License at
	#
	#   http://www.apache.org/licenses/LICENSE-2.0
	#
	#  Unless required by applicable law or agreed to in writing, software
	#  distributed under the License is distributed on an "AS IS" BASIS,
	#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	#  See the License for the specific language governing permissions and
	#  limitations under the License.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def shallow(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 64]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 64]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 14 * 14 * 64]
  pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 14 * 14 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024] # batch_size is the number of training data at the moment
  # Output Tensor Shape: [batch_size, 24]
  logits = tf.layers.dense(inputs=dropout, units=24)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "recall": tf.metrics.recall(
      	  labels=labels, predictions=predictions["classes"]),
      "precision": tf.metrics.precision(
      	  labels=labels, predictions=predictions["classes"])
      }
  
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def deeper(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #2 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #3 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 128]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 128]
  conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * 128])

  # Fully Connected Layer # 1
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 128]
  # Output Tensor Shape: [batch_size, 1024]
  fc1 = tf.layers.dense(inputs=conv3_flat, units=1024, activation=tf.nn.relu)

   # Fully Connected Layer # 2
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 1024]
  fc2 = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu)


  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=fc2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024] # batch_size is the number of training data at the moment
  # Output Tensor Shape: [batch_size, 24]
  logits = tf.layers.dense(inputs=dropout, units=24)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "recall": tf.metrics.recall(
      	  labels=labels, predictions=predictions["classes"]),
      "precision": tf.metrics.precision(
      	  labels=labels, predictions=predictions["classes"])
      }
  
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def multiscale(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Rescale model
  # Resizes input_layer to 4D tensor: [batchsize, 14, 14, 1]
  scaled_layer1 = tf.image.resize_images(images=input_layer, size = [14,14], method = tf.image.ResizeMethod.BICUBIC)

  # Rescale model
  # Resizes input_layer to 4D tensor: [batchsize, 7, 7, 1]
  scaled_layer2 = tf.image.resize_images(images=input_layer, size = [7,7], method = tf.image.ResizeMethod.BICUBIC)

  # Convolutional Layer #1 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14 * 14 * 32]
  pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 32])


  # Convolutional Layer #2 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  conv2 = tf.layers.conv2d(
      inputs=scaled_layer1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7,32]
  # Output Tensor Shape: [batch_size, 7 * 7 * 32]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])

  # Convolutional Layer #3 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 128]
  conv3 = tf.layers.conv2d(
      inputs=scaled_layer2,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 32]
  # Output Tensor Shape: [batch_size, 7 * 7 * 32]
  conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * 32])

  # Combine all the pixels to 2D Tensors
  combined = tf.concat([pool1_flat, pool2_flat, conv3_flat], axis = 1)


  # Fully Connected Layer # 1
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 14*14*7*7*7 * 7 * 32^3]
  # Output Tensor Shape: [batch_size, 1024]
  fc = tf.layers.dense(inputs=combined, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=fc, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024] # batch_size is the number of training data at the moment
  # Output Tensor Shape: [batch_size, 24]
  logits = tf.layers.dense(inputs=dropout, units=24)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "recall": tf.metrics.recall(
      	  labels=labels, predictions=predictions["classes"]),
      "precision": tf.metrics.precision(
      	  labels=labels, predictions=predictions["classes"])
      }
  
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def multiscale2(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14 * 14 * 32]
  pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 32])


  # Convolutional Layer #3
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv3 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #4
  # Computes 32 features uing a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7,32]
  # Output Tensor Shape: [batch_size, 7 * 7 * 32]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Convolutional Layer #3 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 128]
  conv5 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 128]
  conv6 = tf.layers.conv2d(
      inputs=conv5,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 32]
  # Output Tensor Shape: [batch_size, 7 * 7 * 32]
  conv6_flat = tf.reshape(conv6, [-1, 7 * 7 * 128])

  # Combine all the pixels to 2D Tensors
  combined = tf.concat([pool1_flat, pool2_flat, conv6_flat], axis = 1)


  # Fully Connected Layer # 1
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 14*14*7*7*7 * 7 * 32^3]
  # Output Tensor Shape: [batch_size, 1024]
  fc = tf.layers.dense(inputs=combined, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=fc, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024] # batch_size is the number of training data at the moment
  # Output Tensor Shape: [batch_size, 24]
  logits = tf.layers.dense(inputs=dropout, units=24)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "recall": tf.metrics.recall(
      	  labels=labels, predictions=predictions["classes"]),
      "precision": tf.metrics.precision(
      	  labels=labels, predictions=predictions["classes"])
      }
  
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def multiscale3(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 3]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=3,
      kernel_size=[1, 1],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 2]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout1 = tf.layers.dropout(
      inputs=pool1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14 * 14 * 32]
  pool1_flat = tf.reshape(dropout1, [-1, 14 * 14 * 32])

  # Convolutional Layer #3
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv4 = tf.layers.conv2d(
      inputs=dropout1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #4
  # Computes 32 features uing a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
  
  # Add dropout operation; 0.6 probability that element will be kept
  dropout2 = tf.layers.dropout(
      inputs=pool2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)


  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7,32]
  # Output Tensor Shape: [batch_size, 7 * 7 * 32]
  pool2_flat = tf.reshape(dropout2, [-1, 7 * 7 * 64])

  # Convolutional Layer #3 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 128]
  conv6 = tf.layers.conv2d(
      inputs=dropout2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 128]
  conv7 = tf.layers.conv2d(
      inputs=conv6,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 7, 7, 128]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
  
  # Add dropout operation; 0.6 probability that element will be kept
  dropout3 = tf.layers.dropout(
      inputs=pool3, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)


  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 3, 3,128]
  # Output Tensor Shape: [batch_size, 3 * 3 * 128]
  pool3_flat = tf.reshape(dropout3, [-1, 3 * 3 * 128])
  
  # Combine all the pixels to 2D Tensors
  combined = tf.concat([pool1_flat, pool2_flat, pool3_flat], axis = 1)


  # Fully Connected Layer # 1
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 14*14*7*7*7 * 7 * 32^3]
  # Output Tensor Shape: [batch_size, 1024]
  fc1 = tf.layers.dense(inputs=combined, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout_fc1 = tf.layers.dropout(
      inputs=fc1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Fully Connected Layer # 2
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 14*14*7*7*7 * 7 * 32^3]
  # Output Tensor Shape: [batch_size, 1024]
  fc2 = tf.layers.dense(inputs=dropout_fc1, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout_fc2 = tf.layers.dropout(
      inputs=fc2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024] # batch_size is the number of training data at the moment
  # Output Tensor Shape: [batch_size, 24]
  logits = tf.layers.dense(inputs=dropout_fc2, units=24)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "recall": tf.metrics.recall(
      	  labels=labels, predictions=predictions["classes"]),
      "precision": tf.metrics.precision(
      	  labels=labels, predictions=predictions["classes"])
      }
  
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def multiscale4(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 3]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=3,
      kernel_size=[1, 1],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 2]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14 * 14 * 32]
  pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 32])

  # Convolutional Layer #3
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv4 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #4
  # Computes 32 features uing a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7,32]
  # Output Tensor Shape: [batch_size, 7 * 7 * 32]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Convolutional Layer #3 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 128]
  conv6 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3 
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 128]
  conv7 = tf.layers.conv2d(
      inputs=conv6,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 7, 7, 128]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
  

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 3, 3,128]
  # Output Tensor Shape: [batch_size, 3 * 3 * 128]
  pool3_flat = tf.reshape(pool3, [-1, 3 * 3 * 128])
  
  # Combine all the pixels to 2D Tensors
  combined = tf.concat([pool1_flat, pool2_flat, pool3_flat], axis = 1)


  # Fully Connected Layer # 1
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 14*14*7*7*7 * 7 * 32^3]
  # Output Tensor Shape: [batch_size, 1024]
  fc1 = tf.layers.dense(inputs=combined, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout_fc1 = tf.layers.dropout(
      inputs=fc1, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Fully Connected Layer # 2
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 14*14*7*7*7 * 7 * 32^3]
  # Output Tensor Shape: [batch_size, 1024]
  fc2 = tf.layers.dense(inputs=dropout_fc1, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout_fc2 = tf.layers.dropout(
      inputs=fc2, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024] # batch_size is the number of training data at the moment
  # Output Tensor Shape: [batch_size, 24]
  logits = tf.layers.dense(inputs=dropout_fc2, units=24)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "recall": tf.metrics.recall(
      	  labels=labels, predictions=predictions["classes"]),
      "precision": tf.metrics.precision(
      	  labels=labels, predictions=predictions["classes"])
      }
  
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

