# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test for API generation."""

import json

import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core.api import api_gen
from tensorflow_examples.lite.model_maker.core.api import api_util
from tensorflow_examples.lite.model_maker.core.api import include  # pylint: disable=unused-import


class ApiGenTest(tf.test.TestCase):

  def test_golden_api(self):
    golden = api_gen.load_golden(api_gen.DEFAULT_API_FILE)
    imports = api_util.generate_imports(api_util.PACKAGE_PREFIX)

    imports_json = json.dumps(imports, indent=2, sort_keys=True)
    golden_content = api_gen._read_golden_text(api_gen.DEFAULT_API_FILE)
    msg = ('Exported APIs do not match `golden_api.json`. Please check it.\n\n'
           'Imports in json format: \n{}\n\n\n'
           'Golden file content:\n{}\n\n').format(imports_json, golden_content)
    self.assertDictEqual(imports, golden, msg)


if __name__ == '__main__':
  tf.test.main()
