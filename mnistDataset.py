''' This file is to preprocess the images in order to use the CNN
    provided this this folder. This file should run first before
    any other file. 

    The inspiration on this file came from the Tensor Flow Examples:
    https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/contrib/learn/python/learn/datasets/mnist.py
    # Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

    We did not use the code 100% but we got how the input images should be. 

    Note that the input images are from a data set that can be found: 
    https://www.kaggle.com/datamunge/sign-language-mnist/data

    Include in the README folder how to preprocess data
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import csv
import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

'''
    There are 24 classes in our data (The whole alphabet excluding j & z which uses motion)
    The data comes in a CVS. 
'''


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_data(filepath):
  ''' We will extract the Data from the csv file '''
  images = []
  labels = []
  firstline = True

  with open(filepath) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      if firstline:    #skip first line
        firstline = False
        continue

      row_list = list(row)
      images.append(row_list[1:])

      # Relabel labels (since labels 9(J) & 25(Z) don't exist)
      if int(row[0]) > 9:
        labels.append(int(row[0])-1)
      else:
        labels.append(int(row[0]))

  images = np.array(images).astype(int)
  labels = np.array(labels).astype(int)
  
  return images, labels
    


# def extract_labels(f, one_hot = False, num_classes = 5):

class DataSet(object):
  ''' Making the Dataset Class for training data and testing data'''
  def __init__(self,images,labels,dtype = dtypes.float32,reshape = False,seed = None ):
    # If seed is not set, use whatever graph level seed is returned
    seed1, seed2 = random_seed.get_seed(seed)
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype

    if dtype not in (dtypes.uint8, dtypes.float32):
        raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

    assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    
    self._num_examples = images.shape[0]

    # Convert shape from [num explames, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth ==1)
    if reshape:
        # assert images.shape[3] == 1
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

    # Convert from [0, 255] -> [0.0, 1.0]
    if dtype == dtypes.float32:
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0/255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  ''' Next few definitions are to get the variables from the class '''
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

  ''' Return the next batch_size exaamples from this data set. '''
  def next_batch(self, batch_size, shuffle=True):
  
      start = self._index_in_epoch
      # Shuffle for the first epoch
      if self._epochs_completed == 0 and start == 0 and shuffle:
        perm0 = np.arange(self._num_examples)
        np.random.shuffle(perm0)
        self._images = self.images[perm0]
        self._labels = self.labels[perm0]
      # Go to the next epoch
      if start + batch_size > self._num_examples:
        # Finished epoch
        self._epochs_completed += 1
        # Get the rest examples in this epoch
        rest_num_examples = self._num_examples - start
        images_rest_part = self._images[start:self._num_examples]
        labels_rest_part = self._labels[start:self._num_examples]
        # Shuffle the data
        if shuffle:
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._images = self.images[perm]
          self._labels = self.labels[perm]
        # Start next epoch
        start = 0
        self._index_in_epoch = batch_size - rest_num_examples
        end = self._index_in_epoch
        images_new_part = self._images[start:end]
        labels_new_part = self._labels[start:end]
        return np.concatenate(
            (images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
      else:
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]



def load_dataset(one_hot=True,
                   dtype=dtypes.float32,
                   reshape=False, # We make it false , since the csv file already comes reshaped
                   validation_size=0,
                   seed=None):
  class DataSets(object):
    pass
  
  data_sets = DataSets()
  TRAIN = './Data/sign_mnist_train.csv'
  TEST = './Data/sign_mnist_test.csv'
  n_classes = 24

  # Get Training Images and labels
  train_images, train_labels = extract_data(TRAIN)

  # Get Testing Images and labels
  test_images, test_labels = extract_data(TEST)

  if(one_hot):
    train_labels = dense_to_one_hot(train_labels, n_classes)
    test_labels = dense_to_one_hot(test_labels, n_classes)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                   .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]



  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  data_sets.train = DataSet(train_images, train_labels, **options)
  data_sets.validation = DataSet(validation_images, validation_labels, **options)
  data_sets.test = DataSet(test_images, test_labels, **options)
  
  return data_sets

