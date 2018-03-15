from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import mnistDataset
import cnn_models
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as colors

tf.logging.set_verbosity(tf.logging.INFO)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  #     print(cm)
  
  plt.figure(figsize=(20,20))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig('ConfusionMatrix.png')

def main(unused_argv):
  # Load training and eval data
  mnist = mnistDataset.load_dataset(one_hot=False)
  train_data = mnist.train.images  # Returns np.array

  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  print(train_labels.shape)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_models.multiscale4, model_dir="./Models/multiscale4")

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x" : eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  
  prediction = mnist_classifier.predict(input_fn =eval_input_fn)
  
  pred_list = []
  for i in prediction :
    pred_list.append(i['classes'])
  
  # print (pred_list)

  cm = confusion_matrix(eval_labels, np.asarray(pred_list))
  classes = ['A', 'B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

  plt.figure(1)
  plot_confusion_matrix(cm, classes = classes, normalize=False)
  
  # ## convert your array into a dataframe
  # df = pd.DataFrame(cm)

  # ## save to xlsx file
  # filepath = 'confusion_matrix.xlsx'

  # df.to_excel(filepath, index=False)

  



if __name__ == "__main__":
  tf.app.run()