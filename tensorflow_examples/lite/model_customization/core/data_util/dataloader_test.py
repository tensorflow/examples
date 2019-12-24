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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf # TF2
from tensorflow_examples.lite.model_customization.core.data_util import dataloader


class DataLoaderTest(tf.test.TestCase):

  def test_split(self):
    ds = tf.data.Dataset.from_tensor_slices([[0, 1], [1, 1], [0, 0], [1, 0]])
    data = dataloader.DataLoader(ds, 4)
    train_data, test_data = data.split(0.5, shuffle=False)

    self.assertEqual(train_data.size, 2)
    for i, elem in enumerate(train_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 1])).all())

    self.assertEqual(test_data.size, 2)
    for i, elem in enumerate(test_data.dataset):
      self.assertTrue((elem.numpy() == np.array([i, 0])).all())


if __name__ == '__main__':
  assert tf.__version__.startswith('2')
  tf.test.main()
