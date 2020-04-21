# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Custom classification model that is already retained by data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import model_export_format as mef
from tensorflow_examples.lite.model_maker.core.task import custom_model


class ClassificationModel(custom_model.CustomModel):
  """"The abstract base class that represents a Tensorflow classification model."""

  def __init__(self, model_export_format, model_spec, index_to_label,
               num_classes, shuffle, train_whole_model):
    """Initialize a instance with data, deploy mode and other related parameters.

    Args:
      model_export_format: Model export format such as saved_model / tflite.
      model_spec: Specification for the model.
      index_to_label: A list that map from index to label class name.
      num_classes: Number of label classes.
      shuffle: Whether the data should be shuffled.
      train_whole_model: If true, the Hub module is trained together with the
        classification layer on top. Otherwise, only train the top
        classification layer.
    """
    if model_export_format != mef.ModelExportFormat.TFLITE:
      raise ValueError('Model export format %s is not supported currently.' %
                       str(model_export_format))

    super(ClassificationModel, self).__init__(model_export_format, model_spec,
                                              shuffle)
    self.index_to_label = index_to_label
    self.num_classes = num_classes
    self.train_whole_model = train_whole_model

  def evaluate(self, data, batch_size=32):
    """Evaluates the model.

    Args:
      data: Data to be evaluated.
      batch_size: Number of samples per evaluation step.

    Returns:
      The loss value and accuracy.
    """
    ds = self._gen_dataset(data, batch_size, is_training=False)

    return self.model.evaluate(ds)

  def predict_top_k(self, data, k=1, batch_size=32):
    """Predicts the top-k predictions.

    Args:
      data: Data to be evaluated.
      k: Number of top results to be predicted.
      batch_size: Number of samples per evaluation step.

    Returns:
      top k results. Each one is (label, probability).
    """
    if k < 0:
      raise ValueError('K should be equal or larger than 0.')
    ds = self._gen_dataset(data, batch_size, is_training=False)

    predicted_prob = self.model.predict(ds)
    topk_prob, topk_id = tf.math.top_k(predicted_prob, k=k)
    topk_label = np.array(self.index_to_label)[topk_id.numpy()]

    label_prob = []
    for label, prob in zip(topk_label, topk_prob.numpy()):
      label_prob.append(list(zip(label, prob)))

    return label_prob

  def _export_tflite(self,
                     tflite_filename,
                     label_filename,
                     quantized=False,
                     quantization_steps=None,
                     representative_data=None):
    """Converts the retrained model to tflite format and saves it.

    Args:
      tflite_filename: File name to save tflite model.
      label_filename: File name to save labels.
      quantized: boolean, if True, save quantized model.
      quantization_steps: Number of post-training quantization calibration steps
        to run. Used only if `quantized` is True.
      representative_data: Representative data used for post-training
        quantization. Used only if `quantized` is True.
    """
    super(ClassificationModel,
          self)._export_tflite(tflite_filename, quantized, quantization_steps,
                               representative_data)

    with tf.io.gfile.GFile(label_filename, 'w') as f:
      f.write('\n'.join(self.index_to_label))

    tf.compat.v1.logging.info('Saved labels in %s.', label_filename)
