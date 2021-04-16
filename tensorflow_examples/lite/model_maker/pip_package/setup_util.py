# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Util for setup."""
import os
import pathlib
import shutil
import subprocess
import sys

from setuptools import find_namespace_packages


class PackageGen:
  """Organizes package and generates APIs structure."""

  def __init__(
      self,
      base_dir,
      root_dir,
      build_dir,
      nightly,
      lib_ns,
      api_ns,
      internal_name,
  ):
    """Init.

    Args:
      base_dir: Path, path to the base dir. (e.g., Path('model_maker'))
      root_dir: Path, path to the root dir. (e.g., Path('<github repo>'))
      build_dir: Path, path to the build dir, where we copy all source code and
        create a package. (e.g. Path('pip_package/src')).
      nightly: boolean, whether it is nightly.
      lib_ns: str, original namespace of the lib (e.g.
        'tensorflow_examples.lite.model_maker').
      api_ns: str, official package namespace for API. Used as code name. (e.g.,
        'tflite_model_maker').
      internal_name: str, internal package name (e.g., 'python') that creates
        <api_ns>.<internal_name> to map internal packages.
    """
    self.base_dir = base_dir
    self.root_dir = root_dir
    self.build_dir = build_dir
    self.nightly = nightly
    self.lib_ns = lib_ns
    self.api_ns = api_ns
    self.internal_name = internal_name

    # Add the root_dir to the PYTHONPATH (used to look up for api_gen).
    root = str(self.root_dir)
    if root not in sys.path:
      sys.path.insert(0, root)

  def run(self):
    """Generate build folder, and returns packages with dir mapping.

    Returns:
      A dict: extra kwargs for setup.
    """
    from tensorflow_examples.lite.model_maker.core.api import api_util  # pylint: disable=g-import-not-at-top

    # Cleanup if `src` folder exists
    if self.build_dir.exists():
      shutil.rmtree(str(self.build_dir), ignore_errors=True)

    lib_names = api_util.split_name(self.lib_ns)
    lib_pkg = self.build_dir.joinpath(*lib_names)

    # Prepare __init__.py.
    api_util.make_dirs_or_not(lib_pkg)
    for i in range(len(lib_names) + 1):
      dirpath = self.build_dir.joinpath(*lib_names[:i])
      init_file = dirpath.joinpath('__init__.py')
      if not init_file.exists():
        api_util.write_python_file(init_file,
                                   api_util.as_package(lib_names[:i]), None)

    # Copy static files.
    static_files = [
        'README.md', 'RELEASE.md', 'requirements.txt',
        'requirements_nightly.txt'
    ]
    for f in static_files:
      shutil.copy2(
          str(self.base_dir.joinpath(f)), str(self.build_dir.joinpath(f)))

    # Copy .py files.
    files = self.base_dir.rglob(r'*')
    include_extentions = {'.py', '.json'}

    script_pys = []
    init_pys = []
    for path in files:
      name = str(path)
      if path.is_dir():
        continue
      if path.suffix not in include_extentions:
        continue
      if ('pip_package' in name) or ('_test.py' in name):
        continue

      target_path = lib_pkg.joinpath(path.relative_to(self.base_dir))
      api_util.make_dirs_or_not(target_path.parent)
      shutil.copy2(str(path), str(target_path))

      if path.suffix == '.py':
        relative = target_path.relative_to(lib_pkg)
        if path.stem != '__init__':
          # Get a path like: a/b.py
          script_pys.append(relative)
        else:
          # Get a path like: a/__init__.py
          init_pys.append(relative)

    # Create API's namespace mapping.
    internal_names = api_util.split_name(self.api_ns) + api_util.split_name(
        self.internal_name)
    internal_pkg = self.build_dir.joinpath(*internal_names)
    for path in script_pys + init_pys:
      official_py = internal_pkg.joinpath(path)
      if path.stem != '__init__':
        # For example, `a/b.py` becomes `a.b`
        ns = str(path.with_suffix('')).replace(os.sep, '.')
      else:
        # For example, `a/__init__.py` becomes `a`
        ns = str(path.parent).replace(os.sep, '.')

      if ns == '.':  # Replace dir '.' with empty namespace.
        ns = ''
      real_ns = api_util.as_package(
          api_util.split_name(self.lib_ns) + api_util.split_name(ns))
      content = 'from {} import *'.format(real_ns)
      api_util.make_dirs_or_not(official_py.parent)
      api_util.write_python_file(official_py, real_ns, [content])

    # Add APIs files.
    self.add_api_files()

    package_data = {
        '': ['*.txt', '*.md', '*.json'],
    }
    if self.nightly:
      # For nightly, proceeed with addtional preparation.
      extra_package_data = self._prepare_nightly()
      package_data.update(extra_package_data)

    # Return package.
    namespace_packages = find_namespace_packages(where=self.build_dir)

    return {
        'packages': namespace_packages,
        'package_data': package_data,
    }

  def add_api_files(self):
    """Adds API files."""
    from tensorflow_examples.lite.model_maker.core.api import api_gen  # pylint: disable=g-import-not-at-top
    api_gen.run(str(self.build_dir), api_gen.DEFAULT_API_FILE)

  def _prepare_nightly(self):
    """Prepares nightly and gets extra setup config.

    For nightly, tflite-model-maker will pack `tensorflowjs` python source code.

    TODO(tianlin): tensorflowjs pip requires stable tensorflow instead of
    nightly,
    which conflicts with tflite-model-maker-nightly. Thus, we include its python
    code directly.

    Returns:
      dict: extra package_data.
    """
    tfjs_git = 'https://github.com/tensorflow/tfjs'
    tfjs_path = pathlib.Path(__file__).with_name('tfjs')

    # Remove existing tfjs and git clone.
    if tfjs_path.exists():
      shutil.rmtree(str(tfjs_path), ignore_errors=True)
    cmd = ['git', 'clone', tfjs_git, str(tfjs_path)]
    print('Running git clone: {}'.format(cmd))
    subprocess.check_call(cmd)

    # Copy `tensorflowjs` python code, and release with tflite-model-maker
    src_folder = str(
        tfjs_path.joinpath('tfjs-converter', 'python', 'tensorflowjs'))
    dst_folder = str(self.build_dir.joinpath('tensorflowjs'))
    shutil.copytree(
        src_folder,
        dst_folder,
        ignore=lambda _, names: set(s for s in names if s.endswith('_test.py')))

    return {'tensorflowjs/op_list': ['*.json']}
