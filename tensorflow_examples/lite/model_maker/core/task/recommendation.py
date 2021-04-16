# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""APIs to train an on-device recommendation model."""
import collections
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.core.data_util import data_util
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import custom_model
from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
from tensorflow_examples.lite.model_maker.core.task import model_util
from tensorflow_examples.lite.model_maker.core.task.model_spec import recommendation_spec
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import keras_metrics as _metrics
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import recommendation_model_launcher_keras as _launcher


@mm_export('recommendation.create')
def create(train_data,
           model_spec,
           model_spec_options=None,
           model_dir=None,
           validation_data=None,
           batch_size=16,
           steps_per_epoch=10000,
           epochs=1,
           learning_rate=0.1,
           gradient_clip_norm=1.0,
           shuffle=True,
           do_train=True,
           max_history_length=10):
  """Loads data and train the model for recommendation.

  Args:
    train_data: Training data.
    model_spec: Specification for the model.
    model_spec_options: dict, additional options to creat a model.
    model_dir: str, path to export model checkpoints and summaries.
    validation_data: Validation data.
    batch_size: Batch size for training.
    steps_per_epoch: int, Number of step per epoch.
    epochs: int, Number of epochs for training.
    learning_rate: float, learning rate.
    gradient_clip_norm: float, clip threshold (<= 0 meaning no clip).
    shuffle: boolean, whether the training data should be shuffled.
    do_train: boolean, whether to run training.
    max_history_length: int, max history length as model input (for inference).

  Returns:
    object of QuestionAnswer class.
  """
  # Create model spec.
  if model_spec_options is None:
    model_spec_options = {}
  model_spec = ms.get(model_spec)(**model_spec_options)

  # Use model_dir or a temp folder to store intermediate checkpoints, etc.
  if model_dir is None:
    model_dir = tempfile.mkdtemp()

  recommendation = Recommendation(
      model_spec,
      model_dir=model_dir,
      shuffle=shuffle,
      max_history_length=max_history_length,
      learning_rate=learning_rate,
      gradient_clip_norm=gradient_clip_norm)

  if do_train:
    tf.compat.v1.logging.info('Training recommendation model...')
    recommendation.train(
        train_data,
        validation_data,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs)
  else:
    recommendation.create_model(do_train=False)
  return recommendation


@mm_export('recommendation.Recommendation')
class Recommendation(custom_model.CustomModel):
  """Recommendation task class."""

  DEFAULT_EXPORT_FORMAT = (ExportFormat.TFLITE,)
  ALLOWED_EXPORT_FORMAT = (ExportFormat.LABEL, ExportFormat.TFLITE,
                           ExportFormat.SAVED_MODEL)

  # ID = 0 means a placeholder to OOV. Used for padding.
  OOV_ID = 0

  def __init__(self,
               model_spec,
               model_dir,
               shuffle=True,
               max_history_length=10,
               learning_rate=0.1,
               gradient_clip_norm=1.0):
    """Init recommendation model.

    Args:
      model_spec: recommendation model spec.
      model_dir: str, path to export model checkpoints and summaries.
      shuffle: boolean, whether the training data should be shuffled.
      max_history_length: int, max history length as model input for inference.
      learning_rate: float, learning rate.
      gradient_clip_norm: float, clip threshold (<= 0 meaning no clip).
    """
    if not isinstance(model_spec, recommendation_spec.RecommendationSpec):
      raise ValueError(
          'Expect RecommendationSpec but got model_spec: {}'.format(model_spec))
    self._model_dir = model_dir
    self._max_history_length = max_history_length
    self._learning_rate = learning_rate
    self._gradient_clip_norm = gradient_clip_norm
    super(Recommendation, self).__init__(model_spec, shuffle=shuffle)

  def create_model(self, do_train=True):
    """Creates a model.

    Args:
      do_train: boolean. Whether to train the model.

    Returns:
      Keras model.
    """
    self.model = self.model_spec.create_model()
    if do_train:
      _launcher.compile_model(self.model, self.model_spec.params,
                              self._learning_rate, self._gradient_clip_norm)

  def train(self,
            train_data,
            validation_data=None,
            batch_size=16,
            steps_per_epoch=100,
            epochs=1):
    """Feeds the training data for training.

    Args:
      train_data: Training dataset.
      validation_data: Validation data. If None, skips validation process.
      batch_size: int, the batch size.
      steps_per_epoch: int, the step of each epoch.
      epochs: int, number of epochs.

    Returns:
      History from model.fit().
    """
    batch_size = batch_size if batch_size else self.model_spec.batch_size

    train_ds = train_data.gen_dataset(
        batch_size, is_training=True, shuffle=self.shuffle)
    if validation_data:
      validation_ds = validation_data.gen_dataset(batch_size, is_training=False)
    else:
      validation_ds = None

    self.create_model(do_train=True)
    history = self.model.fit(
        x=train_ds,
        validation_data=validation_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=self._keras_callbacks(self._model_dir))
    tf.get_logger().info(history)
    return history

  def evaluate(self, data, batch_size=10):
    """Evaluate the model.

    Args:
      data: Evaluation data.
      batch_size: int, batch size for evaluation.

    Returns:
      History from model.evaluate().
    """
    batch_size = batch_size if batch_size else self.model_spec.batch_size
    eval_ds = data.gen_dataset(batch_size, is_training=False)
    history = self.model.evaluate(eval_ds)
    tf.get_logger().info(history)
    return history

  def _keras_callbacks(self, model_dir):
    """Returns a list of default keras callbacks for `model.fit`."""
    return _launcher.get_callbacks(self.model, model_dir)

  def _export_tflite(self, tflite_filepath):
    """Exports tflite model."""
    # Export TFLite model with max_history_length.
    fn = self.model.serve.get_concrete_function(
        input_context=tf.TensorSpec(
            shape=[self._max_history_length], dtype=tf.int32, name='context'))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([fn])
    tflite_model = converter.convert()
    with tf.io.gfile.GFile(tflite_filepath, 'wb') as f:
      f.write(tflite_model)

  def evaluate_tflite(self, tflite_filepath, data):
    """Evaluates the tflite model.

    The data is padded to required length, and multiple metrics are evaluated.

    Args:
      tflite_filepath: File path to the TFLite model.
      data: Data to be evaluated.

    Returns:
      Dict of (metric, value), evaluation result of TFLite model.
    """
    lite_runner = model_util.get_lite_runner(tflite_filepath, self.model_spec)
    ds = data.gen_dataset(batch_size=1, is_training=False)

    max_output_size = data.max_vocab_id + 1  # +1 because 0 is reserved for OOV.
    eval_top_k = self.model_spec.params['eval_top_k']
    metrics = [
        _metrics.GlobalRecall(name=f'Global_Recall/Recall_{k}', top_k=k)
        for k in eval_top_k
    ]
    for feature, label in data_util.generate_elements(ds):
      context = feature['context'][0].astype(np.int32)
      # x: shape [max_length], padded with PAD_ID
      if len(context) == self._max_history_length:
        x = context
      elif len(context) > self._max_history_length:
        x = context[:self._max_history_length]
      else:
        # Padding.
        x = np.empty([self._max_history_length], dtype=np.int32)
        x.fill(self.OOV_ID)
        x[:len(context)] = context

      # ids, scores: shape [top_k]
      ids, scores = lite_runner.run(x)

      # y_true: shape [1, 1]
      y_true = label
      # y_pred: shape [1, max_output_size]; fill only scores with top-k ids.
      y_pred = np.zeros([1, max_output_size])
      for i, score in zip(ids, scores):
        if i in data.vocab:  # Only set if id is in vocab.
          y_pred[0, i] = score

      # Update metrics.
      for m in metrics:
        m.update_state(y_true, y_pred)
    result = collections.OrderedDict([(m.name, m.result()) for m in metrics])
    return result
