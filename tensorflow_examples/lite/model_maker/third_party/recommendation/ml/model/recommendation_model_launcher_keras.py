# Lint as: python3
#   Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Personalized recommendation model runner based on Tensorflow keras API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import keras_losses as losses
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import keras_metrics as metrics
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import recommendation_model
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import utils

FLAGS = flags.FLAGS

CONTEXT = 'context'
LABEL = 'label'


def define_flags():
  """Define flags."""
  flags.DEFINE_string('training_data_filepattern', None,
                      'File pattern of the training data.')
  flags.DEFINE_string('testing_data_filepattern', None,
                      'File pattern of the training data.')
  flags.DEFINE_string('model_dir', None, 'Directory to store checkpoints.')
  flags.DEFINE_string(
      'params_path', None,
      'Path to the json file containing params needed to run '
      'p13n recommendation model.')
  flags.DEFINE_integer('batch_size', 1, 'Training batch size.')
  flags.DEFINE_float('learning_rate', 0.1, 'Learning rate.')
  flags.DEFINE_integer('steps_per_epoch', 10,
                       'Number of steps to run in each epoch.')
  flags.DEFINE_integer('num_epochs', 10000, 'Number of training epochs.')
  flags.DEFINE_integer('num_eval_steps', 1000, 'Number of eval steps.')
  flags.DEFINE_enum('run_mode', 'train_and_eval', ['train_and_eval', 'export'],
                    'Mode of the launcher, default value is: train_and_eval')
  flags.DEFINE_float('gradient_clip_norm', 1.0,
                     'gradient_clip_norm <= 0 meaning no clip.')
  flags.DEFINE_integer('max_history_length', 10, 'Max length of user history.')
  flags.DEFINE_integer('num_predictions', 100,
                       'Num of top predictions to output.')
  flags.DEFINE_string(
      'encoder_type', 'bow', 'Type of the encoder for context'
      'encoding, the value could be ["bow", "rnn", "cnn"].')
  flags.DEFINE_string('checkpoint_path', '', 'Path to the checkpoint.')


class SimpleCheckpoint(tf.keras.callbacks.Callback):
  """Keras callback to save tf.train.Checkpoints."""

  def __init__(self, checkpoint_manager):
    super(SimpleCheckpoint, self).__init__()
    self.checkpoint_manager = checkpoint_manager

  def on_epoch_end(self, epoch, logs=None):
    step_counter = self.checkpoint_manager._step_counter.numpy()  # pylint: disable=protected-access
    self.checkpoint_manager.save(checkpoint_number=step_counter)


class InputFn:
  """InputFn for recommendation model estimator."""

  def __init__(self,
               data_filepattern,
               batch_size=10,
               shuffle=True,
               repeat=True):
    """Init input fn.

    Args:
      data_filepattern: str, file pattern. (allow wild match like ? and *)
      batch_size: int, batch examples with a given size.
      shuffle: boolean, whether to shuffle examples.
      repeat: boolean, whether to repeat examples.
    """
    self.data_filepattern = data_filepattern
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.repeat = repeat

  @staticmethod
  def decode_example(serialized_proto):
    """Decode single serialized example."""
    name_to_features = dict(
        context=tf.io.VarLenFeature(tf.int64),
        label=tf.io.FixedLenFeature([1], tf.int64))
    record_features = tf.io.parse_single_example(serialized_proto,
                                                 name_to_features)
    for name in record_features:
      t = record_features[name]
      if t.dtype == tf.int64:
        tf.cast(t, tf.int32)
      if isinstance(t, tf.SparseTensor):
        t = tf.sparse.to_dense(t)
      record_features[name] = t
    features = {}
    features['context'] = record_features['context']
    features['label'] = record_features['label']
    return features, record_features['label']

  @staticmethod
  def read_dataset(data_filepattern):
    input_files = utils.GetShardFilenames(data_filepattern)
    return tf.data.TFRecordDataset(input_files)

  def __call__(self):
    """An input_fn satisfying the TF estimator spec.

    Returns:
      a Dataset where each element is a batch of `features` dicts, passed to the
      Estimator model_fn.
    """
    d = self.read_dataset(self.data_filepattern)
    if self.repeat:
      d = d.repeat()
    if self.shuffle:
      buffer_size = max(3 * self.batch_size, 100)
      d = d.shuffle(buffer_size=buffer_size)
    d = d.map(self.decode_example)
    d = d.batch(self.batch_size, drop_remainder=True)
    return d


def _get_optimizer(learning_rate, gradient_clip_norm=None):
  """Gets model optimizer."""
  kwargs = {'clipnorm': gradient_clip_norm} if gradient_clip_norm else {}
  return tf.keras.optimizers.Adagrad(learning_rate, **kwargs)


def _get_metrics(eval_top_k):
  """Gets model evaluation metrics of both batch samples and full vocabulary."""
  metrics_list = [
      metrics.GlobalRecall(name=f'Global_Recall/Recall_{k}', top_k=k)
      for k in eval_top_k
  ]
  metrics_list.append(metrics.GlobalMeanRank(name='global_mean_rank'))
  metrics_list.extend(
      metrics.BatchRecall(name=f'Batch_Recall/Recall_{k}', top_k=k)
      for k in eval_top_k)
  metrics_list.append(metrics.BatchMeanRank(name='batch_mean_rank'))
  return metrics_list


def compile_model(model, params, learning_rate, gradient_clip_norm):
  """Compile keras model."""
  model.compile(
      optimizer=_get_optimizer(
          learning_rate=learning_rate, gradient_clip_norm=gradient_clip_norm),
      loss=losses.GlobalSoftmax(),
      metrics=_get_metrics(params['eval_top_k']))


def build_keras_model(params, learning_rate, gradient_clip_norm):
  """Construct and compile recommendation keras model."""
  model = recommendation_model.RecommendationModel(params)
  compile_model(model, params, learning_rate, gradient_clip_norm)
  return model


def get_callbacks(keras_model, model_dir):
  """Sets up callbacks for training and evaluation."""
  summary_dir = os.path.join(model_dir, 'summaries')
  summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
  checkpoint = tf.train.Checkpoint(
      model=keras_model, optimizer=keras_model.optimizer)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=model_dir,
      max_to_keep=None,
      step_counter=keras_model.optimizer.iterations,
      checkpoint_interval=0)
  checkpoint_callback = SimpleCheckpoint(checkpoint_manager)
  return [summary_callback, checkpoint_callback]


def train_and_eval(model, model_dir, train_input_fn, eval_input_fn,
                   steps_per_epoch, epochs, eval_steps):
  """Train and evaluate."""

  train_dataset = train_input_fn()
  eval_dataset = eval_input_fn()
  callbacks = get_callbacks(model, model_dir)
  history = model.fit(
      x=train_dataset,
      validation_data=eval_dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_steps=eval_steps,
      callbacks=callbacks)
  tf.get_logger().info(history)
  return model


def export(checkpoint_path, export_dir, params, max_history_length):
  """Export savedmodel."""
  model = recommendation_model.RecommendationModel(params)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(checkpoint_path).run_restore_ops()
  signatures = {
      tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          model.serve.get_concrete_function(
              input_context=tf.TensorSpec(
                  shape=[max_history_length],
                  dtype=tf.dtypes.int32,
                  name='context'))
  }
  tf.saved_model.save(model, export_dir=export_dir, signatures=signatures)
  return export_dir


def export_tflite(export_dir):
  """Export to TFLite model.

  Args:
    export_dir: the model exportation dir, where saved_model is located.
  """
  converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
  tflite_model = converter.convert()
  tflite_model_path = os.path.join(export_dir, 'model.tflite')
  with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
    f.write(tflite_model)


def main(_):
  logger = tf.get_logger()
  if not tf.io.gfile.exists(FLAGS.model_dir):
    tf.io.gfile.makedirs(FLAGS.model_dir)

  with tf.io.gfile.GFile(FLAGS.params_path, 'rb') as reader:
    params = json.loads(reader.read().decode('utf-8'))
  params['encoder_type'] = FLAGS.encoder_type
  params['num_predictions'] = FLAGS.num_predictions

  logger.info('Setting up train and eval input_fns.')
  train_input_fn = InputFn(FLAGS.training_data_filepattern, FLAGS.batch_size)
  eval_input_fn = InputFn(FLAGS.testing_data_filepattern, FLAGS.batch_size)

  logger.info('Build keras model for mode: {}.'.format(FLAGS.run_mode))
  model = build_keras_model(params, FLAGS.learning_rate,
                            FLAGS.gradient_clip_norm)

  if FLAGS.run_mode == 'train_and_eval':
    train_and_eval(
        model=model,
        model_dir=FLAGS.model_dir,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        steps_per_epoch=FLAGS.steps_per_epoch,
        epochs=FLAGS.num_epochs,
        eval_steps=FLAGS.num_eval_steps)
  elif FLAGS.run_mode == 'export':
    export_dir = os.path.join(FLAGS.model_dir, 'export')
    logger.info('Exporting model to dir: {}'.format(export_dir))
    export(
        checkpoint_path=FLAGS.checkpoint_path,
        export_dir=export_dir,
        params=params,
        max_history_length=FLAGS.max_history_length)
    logger.info('Converting model to tflite model.')
    export_tflite(export_dir)


if __name__ == '__main__':
  define_flags()
  app.run(main)
