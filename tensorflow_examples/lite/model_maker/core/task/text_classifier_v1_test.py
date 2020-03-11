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

import tensorflow as tf # TF2
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.task import text_classifier_test


class TextClassifierV1Test(text_classifier_test.TextClassifierTest):
  """Share text tests of the base class, but in tf v1 behavior."""


if __name__ == '__main__':
  compat.setup_tf_behavior(tf_version=1)
  tf.test.main()
