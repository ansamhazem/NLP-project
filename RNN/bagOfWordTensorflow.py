# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 01:42:57 2018

@author: Ansam
"""
#[e,t,b,m]
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from sklearn.model_selection import train_test_split
import commans

FLAGS = None

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 50
n_words = 0
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.


def estimator_spec_for_softmax_classification(logits, labels, mode):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy':
          tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def bag_of_words_model(features, labels, mode):
  """A bag-of-words model. Note it disregards the word order in the text."""
  bow_column = tf.feature_column.categorical_column_with_identity(
      WORDS_FEATURE, num_buckets=n_words)
  bow_embedding_column = tf.feature_column.embedding_column(
      bow_column, dimension=EMBEDDING_SIZE)
  bow = tf.feature_column.input_layer(
      features, feature_columns=[bow_embedding_column])
  logits = tf.layers.dense(bow, MAX_LABEL, activation=None)

  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)


def rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""
  #Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  print(features[WORDS_FEATURE])
  word_vectors = tf.contrib.layers.embed_sequence(
      features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = tf.unstack(word_vectors, axis=1)

  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  # Given encoding of RNN, take encoding of last step (e.g hidden size of the
  # neural network of last step) and pass it as features for softmax
  # classification over output classes.
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)


def main(unused_argv):
  global n_words
  tf.logging.set_verbosity(tf.logging.INFO)

  # Prepare training and testing data
  news = pd.read_csv("../data/uci-news-aggregator.csv") #Importing data from CSV
  news = (news.loc[news['CATEGORY'].isin(['b','e','t','m'])]) #Retaining rows that belong to categories 'b' and 'e'
  news=news[0:26000]
  X_train, X_test, Y_train, Y_test = train_test_split(news["TITLE"], news["CATEGORY"], test_size = 0.2)
  x_train = np.array(X_train);
  x_test = np.array(X_test);
  Y_train = np.array(Y_train);
  Y_test = np.array(Y_test);
  print(Y_train)
  y_train=[]
  y_test=[]
  #[e,t,b,m]
  for y in Y_test:
      if(y=='m'): 
#          label=np.zeros(4)
#          label[3] = 1.0
          y_test.append(3)
      if(y=='e'): 
#          label=np.zeros(4)
#          label[0] = 1.0
          y_test.append(0)
      if(y=='t'): 
#          label=np.zeros(4)
#          label[1] = 1.0
          y_test.append(1)
      if(y=='b'): 
#          label=np.zeros(4)
#          label[2] = 1.0
          y_test.append(2)
  print(y_test)
  for y in Y_train:
      if(y=='m'): 
#          label=np.zeros(4)
#          label[3] = 1.0
          y_train.append(3)
      if(y=='e'): 
#          label=np.zeros(4)
#          label[0] = 1.0
          y_train.append(0)
      if(y=='t'): 
#          label=np.zeros(4)
#          label[1] = 1.0
          y_train.append(1)
      if(y=='b'): 
#          label=np.zeros(4)
#          label[2] = 1.0
          y_train.append(2)
  print(y_train)
  y_train=np.array(y_train)  
  y_test=np.array(y_test)     

  # Process vocabulary
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)
  pickle.dump(vocab_processor, open('vocab.pickle', 'wb'))

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  n_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % n_words)

  # Build model
  # Switch between rnn_model and bag_of_words_model to test different models.
  model_fn = rnn_model
  if FLAGS.bow_model:
    # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
    # ids start from 1 and 0 means 'no word'. But
    # categorical_column_with_identity assumes 0-based count and uses -1 for
    # missing word.
    x_train -= 1
    x_test -= 1
    model_fn = bag_of_words_model
  classifier = tf.estimator.Estimator(model_fn=model_fn,model_dir="checkpointBOW/")

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_train},
      y=y_train,
      batch_size=len(x_train),
      num_epochs=None,
      shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=100)

  # Predict.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)
# Score with sklearn.
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

def serving_fn():
    receiver_tensor = {
        commans.FEATURE_COL: tf.placeholder(dtype=tf.string, shape=None)
    }

    _features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(_features, receiver_tensor)
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test_with_fake_data',
      default=False,
      help='Test the example code with fake data.',
      action='store_true')
  parser.add_argument(
      '--bow_model',
      default=True,
      help='Run with BOW model instead of RNN.',
      action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)