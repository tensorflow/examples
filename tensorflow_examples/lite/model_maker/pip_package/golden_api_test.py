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
"""Test all golden APIs.

Run this test, after `tflite_model_maker` package is installed.

python golden_api_test.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from typing import List

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_examples.lite.model_maker.core.api import api_gen

GOLDEN = api_gen.load_golden(api_gen.DEFAULT_API_FILE)


class GoldenApiTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(GOLDEN.items())
  def test_golden_apis(self, package: str, import_lines: List[str]):
    """Test all golden API symbols."""
    import tflite_model_maker  # pylint: disable=g-import-not-at-top

    for line in import_lines:
      # Get `question_answer`, for `tflite_model_maker.question_answer`.
      parts = package.split('.')[1:]
      # Get `c`, for `from a import c` or `from a import b as c`.
      name = line.split()[-1]
      parts.append(name)

      # Assert there is no error for golden API.
      symbol = tflite_model_maker
      for p in parts:
        symbol = getattr(symbol, p)

  @parameterized.parameters(
      # Golden APIs:
      ('tflite_model_maker'),
      ('tflite_model_maker.question_answer'),
      # Internal packages:
      ('tflite_model_maker.python'),
      ('tflite_model_maker.python.cli.cli'),
      ('tflite_model_maker.python.core.export_format'),
      ('tflite_model_maker.python.demo.question_answer_demo'),
  )
  def test_absolute_import(self, name: str):
    """Tests absolute import (internal python.* are subject to change)."""
    importlib.import_module(name)


if __name__ == '__main__':
  tf.test.main()
