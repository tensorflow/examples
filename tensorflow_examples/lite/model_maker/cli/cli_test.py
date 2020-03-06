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

import sys
from unittest.mock import patch

from absl.testing import absltest
from absl.testing import parameterized
import fire

from tensorflow_examples.lite.model_maker.cli import cli
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.demo import image_classification_demo
from tensorflow_examples.lite.model_maker.demo import text_classification_demo

BIN = 'model_maker'  # Whatever binary name.


def patch_image():
  """Patch image classification demo run."""
  return patch.object(image_classification_demo, 'run')


def patch_text():
  """Patch text classification demo run."""
  return patch.object(text_classification_demo, 'run')


def patch_setup():
  """Patch image classification demo run."""
  return patch.object(compat, 'setup_tf_behavior')


class CLITest(parameterized.TestCase):

  @parameterized.parameters(
      (['--tf=1'], 1),
      (['--tf=2'], 2),
      ([], 2),  # No extra flag, default
  )
  def test_init(self, tf_opt, expected_tf):
    sys.argv = [BIN, 'image_classification', 'data', 'lite', 'label'] + tf_opt
    with patch_image() as run, patch_setup() as setup:
      cli.main()
      setup.assert_called_once_with(expected_tf)
      run.assert_called_once_with('data', 'lite', 'label', 'efficientnet_b0')

  @parameterized.parameters(
      ([], ['efficientnet_b0'], {}),
      (['--spec=mobilenet_v2'], ['mobilenet_v2'], {}),
      (['--spec=mobilenet_v2', '--epochs=1'], ['mobilenet_v2'], dict(epochs=1)),
  )
  def test_image_classification_demo(self, opt, args, kwargs):
    sys.argv = [BIN, 'image_classification', 'data', 'lite', 'label'] + opt
    with patch_image() as run:
      cli.main()
      run.assert_called_once_with('data', 'lite', 'label', *args, **kwargs)

  def test_image_classification_demo_lack_param(self):
    sys.argv = [BIN, 'image_classification', 'data', 'lite']
    with patch_image() as run:
      with self.assertRaisesRegex(fire.core.FireExit, '2'):
        cli.main()
      run.assert_not_called()

  @parameterized.parameters(
      ([], ['bert'], {}),
      (['--spec=average_word_vec'], ['average_word_vec'], {}),
      (['--epochs=1'], ['bert'], dict(epochs=1)),
  )
  def test_text_classification_demo(self, opt, args, kwargs):
    sys.argv = [BIN, 'text_classification', 'data', 'lite', 'label', 'vocab'
               ] + opt
    with patch_text() as run:
      cli.main()
      run.assert_called_once_with('data', 'lite', 'label', 'vocab', *args,
                                  **kwargs)

  def test_text_classification_demo_lack_param(self):
    sys.argv = [BIN, 'text_classification', 'data', 'lite', 'label']
    with patch_text() as run:
      with self.assertRaisesRegex(fire.core.FireExit, '2'):
        cli.main()
      run.assert_not_called()

  def test_invalid_command(self):
    sys.argv = [BIN, 'invalid_command']
    with self.assertRaisesRegex(fire.core.FireExit, '2'):
      cli.main()

  @parameterized.parameters(
      ([BIN, '--', '--help'],),
      ([BIN, 'image_classification', '--', '--help'],),
      ([BIN, 'text_classification', '--', '--help'],),
  )
  def test_help(self, opt):
    sys.argv = opt
    with self.assertRaisesRegex(fire.core.FireExit, '0'):
      cli.main()


if __name__ == '__main__':
  absltest.main()
