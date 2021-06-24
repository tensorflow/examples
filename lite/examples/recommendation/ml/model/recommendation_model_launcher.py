# Lint as: python3
#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
import os
import time
from typing import List

from absl import app
from absl import flags
import tensorflow as tf
from configs import input_config_generated_pb2 as input_config_pb2
from configs import model_config as model_config_class
from model import input_pipeline
from model import losses
from model import metrics
from model import recommendation_model

from google.protobuf import text_format


FLAGS = flags.FLAGS


def define_flags():
  """Define flags."""
  flags.DEFINE_string('training_data_filepattern', None,
                      'File pattern of the training data.')
  flags.DEFINE_string('testing_data_filepattern', None,
                      'File pattern of the training data.')
  flags.DEFINE_string('model_dir', None, 'Directory to store checkpoints.')
  flags.DEFINE_string('export_dir', None, 'Directory for the exported model.')
  flags.DEFINE_integer('batch_size', 1, 'Training batch size.')
  flags.DEFINE_float('learning_rate', 0.1, 'Learning rate.')
  flags.DEFINE_integer('steps_per_epoch', 10,
                       'Number of steps to run in each epoch.')
  flags.DEFINE_integer('num_epochs', 10000, 'Number of training epochs.')
  flags.DEFINE_integer('num_eval_steps', 1000, 'Number of eval steps.')
  flags.DEFINE_enum('run_mode', 'train_and_eval',
                    ['train_and_eval', 'export', 'export_tflite'],
                    'Mode of the launcher, default value is: train_and_eval')
  flags.DEFINE_float('gradient_clip_norm', 1.0,
                     'gradient_clip_norm <= 0 meaning no clip.')
  flags.DEFINE_string('vocab_dir', None,
                      'Path of the directory storing vocabulary files.')
  flags.DEFINE_string('input_config_file', None,
                      'Path to the input config pbtxt'
                      'file.')
  flags.DEFINE_list('hidden_layer_dims', None, 'Hidden layer dimensions.')
  flags.DEFINE_list('eval_top_k', None, 'Top k to evaluate.')
  flags.DEFINE_list(
      'conv_num_filter_ratios', None,
      'Number of filter ratios for the Conv1D layer, this'
      'flag is only required if CNN encoder type is used.')
  flags.DEFINE_integer(
      'conv_kernel_size', 4,
      'Size of the Conv1D layer kernel size, this flag is only'
      'required if CNN encoder type is used.')
  flags.DEFINE_integer(
      'lstm_num_units', 4, 'Number of units for the LSTM layer,'
      'this flag is only required if LSTM encoder type is used.')
  flags.DEFINE_integer('num_predictions', 5,
                       'Num of top predictions to output.')
  flags.DEFINE_string('checkpoint_path', '', 'Path to the checkpoint.')


class SimpleCheckpoint(tf.keras.callbacks.Callback):
  """Keras callback to save tf.train.Checkpoints."""

  def __init__(self, checkpoint_manager):
    super(SimpleCheckpoint, self).__init__()
    self.checkpoint_manager = checkpoint_manager

  def on_epoch_end(self, epoch, logs=None):
    step_counter = self.checkpoint_manager._step_counter.numpy()  # pylint: disable=protected-access
    self.checkpoint_manager.save(checkpoint_number=step_counter)


def _get_optimizer(learning_rate: float, gradient_clip_norm: float):
  """Gets model optimizer."""
  kwargs = {'clipnorm': gradient_clip_norm} if gradient_clip_norm > 0 else {}
  return tf.keras.optimizers.Adagrad(learning_rate, **kwargs)


def _get_metrics(eval_top_k: List[int]):
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


def compile_model(model, eval_top_k, learning_rate, gradient_clip_norm):
  """Compile keras model."""
  model.compile(
      optimizer=_get_optimizer(
          learning_rate=learning_rate, gradient_clip_norm=gradient_clip_norm),
      loss=losses.GlobalSoftmax(),
      metrics=_get_metrics(eval_top_k))


def build_keras_model(input_config: input_config_pb2.InputConfig,
                      model_config: model_config_class.ModelConfig):
  """Construct and compile recommendation keras model.

  Construct recommendation model according to input config and model config.
  Compile the model with optimizer, loss function and eval metrics.

  Args:
    input_config: The configuration object(input_config_pb2.InputConfig) that
      holds parameters for model input feature processing.
    model_config: A ModelConfig object that holds parameters to set up the
      model architecture.

  Returns:
    The compiled keras model.
  """
  model = recommendation_model.RecommendationModel(
      input_config=input_config, model_config=model_config)
  compile_model(model, model_config.eval_top_k, FLAGS.learning_rate,
                FLAGS.gradient_clip_norm)
  return model


def get_callbacks(keras_model: tf.keras.Model,
                  model_dir: str):
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


def train_and_eval(model: tf.keras.Model,
                   model_dir: str,
                   train_input_dataset: tf.data.Dataset,
                   eval_input_dataset: tf.data.Dataset,
                   steps_per_epoch: int,
                   epochs: int,
                   eval_steps: int):
  """Train and evaluate."""
  callbacks = get_callbacks(model, model_dir)
  history = model.fit(
      x=train_input_dataset,
      validation_data=eval_input_dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_steps=eval_steps,
      callbacks=callbacks)
  tf.get_logger().info(history)
  return model


def save_model(checkpoint_path: str, export_dir: str,
               input_config: input_config_pb2.InputConfig,
               model_config: model_config_class.ModelConfig):
  """Export to savedmodel.

  Args:
    checkpoint_path: The path to the checkpoint that the model will be exported
      based on.
    export_dir: The directory to export models to.
    input_config: The input config of the model.
    model_config: The configuration to set up the model.
  """
  model = recommendation_model.RecommendationModel(
      input_config=input_config,
      model_config=model_config)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(checkpoint_path).run_restore_ops()
  input_specs = input_pipeline.get_serving_input_specs(input_config)
  signatures = {
      tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          model.serve.get_concrete_function(**input_specs)
  }
  tf.saved_model.save(model, export_dir=export_dir, signatures=signatures)


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


def export(checkpoint_path: str, input_config: input_config_pb2.InputConfig,
           model_config: model_config_class.ModelConfig, export_dir: str):
  """Export to tensorflow saved model and TFLite model.

  Args:
    checkpoint_path: The path to the checkpoint that the model will be exported
      based on.
    input_config: The input config of the model.
    model_config: The configuration to set up the model.
    export_dir: The directory to store the exported model, If not set, model is
      exported to the model_dir with timestamp.
  """
  logger = tf.get_logger()
  if not export_dir:
    export_dir = os.path.join(FLAGS.model_dir, 'export', str(int(time.time())))
  logger.info('Exporting model to dir: {}'.format(export_dir))
  save_model(
      checkpoint_path=checkpoint_path,
      export_dir=export_dir,
      input_config=input_config,
      model_config=model_config)
  logger.info('Converting model to tflite model.')
  export_tflite(export_dir)


def load_input_config():
  """Load input config."""
  assert FLAGS.input_config_file, 'input_config_file cannot be empty.'
  with tf.io.gfile.GFile(FLAGS.input_config_file, 'rb') as reader:
    return text_format.Parse(reader.read(), input_config_pb2.InputConfig())


def prepare_model_config():
  """Prepare model config."""
  return model_config_class.ModelConfig(
      hidden_layer_dims=[int(x) for x in FLAGS.hidden_layer_dims],
      eval_top_k=[int(x) for x in FLAGS.eval_top_k],
      conv_num_filter_ratios=[int(x) for x in FLAGS.conv_num_filter_ratios],
      conv_kernel_size=FLAGS.conv_kernel_size,
      lstm_num_units=FLAGS.lstm_num_units,
      num_predictions=FLAGS.num_predictions)


def main(_):
  logger = tf.get_logger()
  if not tf.io.gfile.exists(FLAGS.model_dir):
    tf.io.gfile.mkdir(FLAGS.model_dir)

  if not tf.io.gfile.exists(FLAGS.export_dir):
    tf.io.gfile.mkdir(FLAGS.export_dir)

  input_config = load_input_config()
  model_config = prepare_model_config()

  logger.info('Setting up train and eval input datasets.')
  train_input_dataset = input_pipeline.get_input_dataset(
      data_filepattern=FLAGS.training_data_filepattern,
      input_config=input_config,
      vocab_file_dir=FLAGS.vocab_dir,
      batch_size=FLAGS.batch_size)
  eval_input_dataset = input_pipeline.get_input_dataset(
      data_filepattern=FLAGS.testing_data_filepattern,
      input_config=input_config,
      vocab_file_dir=FLAGS.vocab_dir,
      batch_size=FLAGS.batch_size)

  logger.info('Build keras model for mode: {}.'.format(FLAGS.run_mode))
  model = build_keras_model(
      input_config=input_config, model_config=model_config)

  if FLAGS.run_mode == 'train_and_eval':
    train_and_eval(
        model=model,
        model_dir=FLAGS.model_dir,
        train_input_dataset=train_input_dataset,
        eval_input_dataset=eval_input_dataset,
        steps_per_epoch=FLAGS.steps_per_epoch,
        epochs=FLAGS.num_epochs,
        eval_steps=FLAGS.num_eval_steps)
    latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    if latest_checkpoint_path:
      export(
          checkpoint_path=latest_checkpoint_path,
          input_config=input_config,
          model_config=model_config,
          export_dir=FLAGS.export_dir)
  elif FLAGS.run_mode == 'export':
    checkpoint_path = (
        FLAGS.checkpoint_path if FLAGS.checkpoint_path else
        tf.train.latest_checkpoint(FLAGS.model_dir))
    export(
        checkpoint_path=checkpoint_path,
        input_config=input_config,
        model_config=model_config,
        export_dir=FLAGS.export_dir)
  else:
    logger.error('Unsupported launcher run model {}.'.format(FLAGS.run_mode))


if __name__ == '__main__':
  define_flags()
  app.run(main)
