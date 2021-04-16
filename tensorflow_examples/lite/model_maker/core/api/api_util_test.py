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
"""Tests for api_util."""

import os
import tempfile

import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core.api import api_util


class MMExportTest(tf.test.TestCase):

  def setUp(self):
    super(MMExportTest, self).setUp()
    api_util._reset_apis()

  def test_call(self):

    @api_util.mm_export('foo.a')
    def a():
      pass

    self.assertLen(api_util.NAME_TO_SYMBOL, 1)
    expected = api_util.Symbol('foo.a', ['foo', 'a'], a, '__main__', 'a')
    self.assertEqual(api_util.NAME_TO_SYMBOL['foo.a'], expected)

  def test_call_class(self):

    @api_util.mm_export('foo.A')
    class A():
      pass

    self.assertLen(api_util.NAME_TO_SYMBOL, 1)
    expected = api_util.Symbol('foo.A', ['foo', 'A'], A, '__main__', 'A')
    self.assertEqual(api_util.NAME_TO_SYMBOL['foo.A'], expected)

  def test_call_duplicated(self):
    with self.assertRaisesRegex(ValueError, 'API already exists'):

      @api_util.mm_export('foo.a')
      def a():  # pylint: disable=unused-variable
        pass

      @api_util.mm_export('foo.a')
      def b():  # pylint: disable=unused-variable
        pass

  def test_call_global_function(self):

    def test_func():
      """Func to test export."""
      pass

    exportor = api_util.mm_export('foo.bar.test_func')
    ret_func = exportor(test_func)
    self.assertEqual(ret_func, test_func)
    func = api_util.NAME_TO_SYMBOL['foo.bar.test_func']
    self.assertEqual(func.gen_import(), 'from __main__ import test_func')
    self.assertEqual(func.get_package_name(), 'foo.bar')

    exportor = api_util.mm_export('fn')
    exportor(test_func)
    func = api_util.NAME_TO_SYMBOL['fn']
    self.assertEqual(func.gen_import(), 'from __main__ import test_func as fn')
    self.assertEqual(func.get_package_name(), '')

  def test_export_constant(self):

    FOO = 1  # pylint: disable=invalid-name,unused-variable
    api_util.mm_export('foo.FOO').export_constant(__name__, 'FOO')

    self.assertLen(api_util.NAME_TO_SYMBOL, 1)
    expected = api_util.Symbol('foo.FOO', ['foo', 'FOO'], None, '__main__',
                               'FOO')
    self.assertEqual(api_util.NAME_TO_SYMBOL['foo.FOO'], expected)


class ApiUtilTest(tf.test.TestCase):

  def setUp(self):
    super(ApiUtilTest, self).setUp()
    api_util._reset_apis()

  def test_get_module_and_name(self):

    def a():
      pass

    module_and_name = api_util._get_module_and_name(a)
    self.assertTupleEqual(module_and_name, ('__main__', 'a'))

    expected = (
        'tensorflow_examples.lite.model_maker.core.api.api_util',
        'generate_imports',
    )
    module_and_name = api_util._get_module_and_name(api_util.generate_imports)
    self.assertTupleEqual(module_and_name, expected)

    class A:
      pass

    module_and_name = api_util._get_module_and_name(A)
    self.assertTupleEqual(module_and_name, ('__main__', 'A'))

  def test_generate_imports(self):

    @api_util.mm_export('foo.a')
    def a():  # pylint: disable=unused-variable
      pass

    @api_util.mm_export('foo.b')
    def b():  # pylint: disable=unused-variable
      pass

    @api_util.mm_export('bar.sub.c')
    def c():  # pylint: disable=unused-variable
      pass

    @api_util.mm_export('bar.sub.aaa')
    def aa():  # pylint: disable=unused-variable
      pass

    imports = api_util.generate_imports('pkg1.pkg2')
    expected = {
        'pkg1.pkg2': [
            'from pkg1.pkg2 import bar',
            'from pkg1.pkg2 import foo',
        ],
        'pkg1.pkg2.foo': [
            'from __main__ import a',
            'from __main__ import b',
        ],
        'pkg1.pkg2.bar': ['from pkg1.pkg2.bar import sub',],
        'pkg1.pkg2.bar.sub': [
            'from __main__ import aa as aaa',
            'from __main__ import c',
        ],
    }
    self.assertDictEqual(imports, expected)

  def test_write_packages(self):

    @api_util.mm_export('foo.a')
    def a():  # pylint: disable=unused-variable
      pass

    @api_util.mm_export('bar.sub.b')
    def b():  # pylint: disable=unused-variable
      pass

    imports = api_util.generate_imports(api_util.PACKAGE_PREFIX)
    expected_imports = {
        'tflite_model_maker': [
            'from tflite_model_maker import bar',
            'from tflite_model_maker import foo',
        ],
        'tflite_model_maker.foo': ['from __main__ import a',],
        'tflite_model_maker.bar': ['from tflite_model_maker.bar import sub',],
        'tflite_model_maker.bar.sub': ['from __main__ import b',],
    }
    self.assertDictEqual(imports, expected_imports)

    with tempfile.TemporaryDirectory() as tmp_dir:
      api_util.write_packages(tmp_dir, imports)

      # Checks existence of __init__ file and its content.
      for package_name, symbols in expected_imports.items():
        parts = api_util.split_name(package_name)
        path = api_util.as_path(parts)
        init_file = os.path.join(tmp_dir, path, '__init__.py')
        self.assertTrue(os.path.exists(init_file))
        self.assertGreater(os.path.getsize(init_file), 0)
        with tf.io.gfile.GFile(init_file) as f:
          content = f.read()
          self.assertIn(api_util.LICENSE, content)
          self.assertIn('Generated API for package:', content)
          for symbol in symbols:
            self.assertIn(symbol, content)


if __name__ == '__main__':
  tf.test.main()
