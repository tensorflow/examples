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
"""Image classification demo code of model customization for TFLite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf # TF2
from tensorflow_examples.lite.model_customization.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_customization.core.model_export_format import ModelExportFormat
from tensorflow_examples.lite.model_customization.core.task import image_classifier
from tensorflow_examples.lite.model_customization.core.task.model_spec import efficientnet_b0_spec

flags.DEFINE_string('tflite_filename', None, 'File name to save tflite model.')
flags.DEFINE_string('label_filename', None, 'File name to save labels.')
FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)

  image_path = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)
  data = ImageClassifierDataLoader.from_folder(image_path)
  train_data, rest_data = data.split(0.8)
  validation_data, test_data = rest_data.split(0.5)

  model = image_classifier.create(
      train_data,
      model_export_format=ModelExportFormat.TFLITE,
      model_spec=efficientnet_b0_spec,
      validation_data=validation_data)

  _, acc = model.evaluate(test_data)
  print('Test accuracy: %f' % acc)

  model.export(FLAGS.tflite_filename, FLAGS.label_filename)


if __name__ == '__main__':
  assert tf.__version__.startswith('2')
  flags.mark_flag_as_required('tflite_filename')
  flags.mark_flag_as_required('label_filename')
  app.run(main)
