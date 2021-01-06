# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
import tempfile

import tensorflow as tf

from tensorflow_examples.lite.model_maker.demo import custom_model_demo


class DemoTest(tf.test.TestCase):

  def test_demo(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      tflite_filename = os.path.join(temp_dir, 'model.tflite')
      saved_model_filename = os.path.join(temp_dir,
                                          'saved_model/saved_model.pb')

      seed = 100
      tf.random.set_seed(seed)
      acc = custom_model_demo.train_xor_model(temp_dir)
      self.assertEqual(acc, 1.)

      def exists(filename):
        self.assertTrue(tf.io.gfile.exists(filename))
        self.assertGreater(os.path.getsize(filename), 0)

      exists(tflite_filename)
      exists(saved_model_filename)


if __name__ == '__main__':
  tf.test.main()
