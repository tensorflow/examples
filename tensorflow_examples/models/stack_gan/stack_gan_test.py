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
# ==============================================================================
"""StackGAN tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import tensorflow as tf # TF2
from tensorflow_examples.models.stack_gan import stack_gan

class StackGANTest(tf.test.TestCase):

  def test_one_epoch(self):
    epochs = 1
    batch_size = 1
    enable_function = True

    input_image = tf.random.uniform((28, 28, 1))
    label = tf.zeros((1,))
    train_dataset = tf.data.Dataset.from_tensors(
        (input_image, label)).batch(batch_size)
    checkpoint_pr = dcgan.get_checkpoint_prefix()
