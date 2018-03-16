'''
   Making the Estimators to train and test. 

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import mnistDataset
import cnn_models

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  # Load training and eval data
  mnist = mnistDataset.load_dataset(one_hot=False)
  train_data = mnist.train.images  # Returns np.array

  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  print(train_labels.shape)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  # mnist_classifier = tf.estimator.Estimator(
  #     model_fn=cnn_models.shallow, model_dir="./Models/shallow")

  # mnist_classifier = tf.estimator.Estimator(
  #     model_fn=cnn_models.deeper, model_dir="./Models/deeper")

  # mnist_classifier = tf.estimator.Estimator(
  #     model_fn=cnn_models.multiscale2, model_dir="./Models/multiscale2")

  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_models.multiscale3, model_dir="./Models/multiscale5")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=128,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps= 20000, # Model will train for 20000 steps total
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()