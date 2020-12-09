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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.data_util import audio_dataloader
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import audio_classifier
from tensorflow_examples.lite.model_maker.core.task.model_spec import audio_spec


class AudioClassifierTest(tf.test.TestCase):

  def testBrowserFFT(self):

    def pcm(shape):
      # Convert random number between (0, 1] to int16
      return np.random.rand(*shape) * (1 << 15)

    np.random.seed(123)

    spec = audio_spec.BrowserFFTSpec()
    dataset_shape = (1, spec.expected_waveform_len)
    sounds = [pcm(dataset_shape) for category in range(2)]
    labels = list(range(2))
    index_to_labels = ['sound1', 'sound2']
    ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(sounds),
                              tf.data.Dataset.from_tensor_slices(labels)))
    ds = ds.map(spec.preprocess)
    data_loader = audio_dataloader.DataLoader(ds, len(ds), index_to_labels)

    task = audio_classifier.create(data_loader, spec, batch_size=1, epochs=100)

    _, acc = task.evaluate(data_loader)
    # Better than random guessing.
    self.assertGreater(acc, .5)

    # Export the model to saved model.
    output_path = os.path.join(spec.model_dir, 'saved_model')
    task.export(spec.model_dir, export_format=ExportFormat.SAVED_MODEL)
    self.assertTrue(os.path.isdir(output_path))
    self.assertNotEqual(len(os.listdir(output_path)), 0)

    # Export the model to TFLite.
    output_path = os.path.join(spec.model_dir, 'float.tflite')
    task.export(
        spec.model_dir,
        tflite_filename='float.tflite',
        export_format=ExportFormat.TFLITE)
    self.assertTrue(tf.io.gfile.exists(output_path))
    self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == '__main__':
  # Load compressed models from tensorflow_hub
  os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
  compat.setup_tf_behavior(tf_version=2)
  tf.test.main()
