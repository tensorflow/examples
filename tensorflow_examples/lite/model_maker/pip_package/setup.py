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
"""Setup for Model Maker.

To build package:
  python setup.py sdist bdist_wheel

To install directly:
  pip install -e .

To uninstall:
  pip uninstall tflite-model-maker
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import pathlib
import shutil
import sys

from setuptools import find_namespace_packages
from setuptools import setup

nightly = False
if '--nightly' in sys.argv:
  nightly = True
  sys.argv.remove('--nightly')

project_name = 'tflite-model-maker'
datestring = datetime.datetime.now().strftime('%Y%m%d%H%M')
classifiers = [
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

if nightly:
  project_name = '{}-nightly'.format(project_name)
  version = '0.1.3'  # Version prefix, usually major.minor.micro.
  version = '{:s}.dev{:s}'.format(version, datestring)
  classifiers += [
      'Development Status :: 4 - Beta',
  ]
else:
  version = '0.1.2'

# Path to folder model_maker.
BASE_DIR = pathlib.Path(os.path.abspath(__file__)).parents[1]
LIB_NAMESPACE = 'tensorflow_examples.lite.model_maker'  # Original namespace.
OFFICIAL_NAMESPACE = 'tflite_model_maker'  # Official package namespace.
MODEL_MAKER_CONSOLE = 'tflite_model_maker=tflite_model_maker.cli.cli:main'
PIP_PKG_PATH = BASE_DIR.joinpath('pip_package')
SRC_NAME = 'src'  # To create folder `pip_package/src`

DESCRIPTION = ('TFLite Model Maker: a model customization library for on-device'
               ' applications.')
with BASE_DIR.joinpath('README.md').open() as readme_file:
  LONG_DESCRIPTION = readme_file.read()


def get_required_packages():
  """Gets packages inside requirements.txt."""
  with BASE_DIR.joinpath('requirements.txt').open() as f:
    required_pkgs = [l.strip() for l in f.read().splitlines()]
    required_pkgs = list(
        filter(lambda line: line and not line.startswith('#'), required_pkgs))
    return required_pkgs


def _ensure_dir_created(dirpath):
  """Ensures dir created."""
  if not dirpath.exists():
    os.makedirs(str(dirpath))


def _create_py_with_content(filepath, content):
  """Creates a py file with content and header of license."""
  with PIP_PKG_PATH.joinpath('header.txt').open('r') as template:
    header = template.read()
  with filepath.open('w') as f:
    f.write(header)
    if content:
      f.write(content)


def prepare_package_src():
  """Prepares src folder, and returns packages with dir mapping."""
  lib_names = LIB_NAMESPACE.split('.')
  build_root = PIP_PKG_PATH.joinpath(SRC_NAME)
  lib_pkg = build_root.joinpath(*lib_names)

  # Cleanup
  if build_root.exists():
    shutil.rmtree(str(build_root), ignore_errors=True)

  # Prepare __init__.py.
  _ensure_dir_created(lib_pkg)
  for i in range(len(lib_names) + 1):
    dirpath = build_root.joinpath(*lib_names[:i])
    init_file = dirpath.joinpath('__init__.py')
    if not init_file.exists():
      _create_py_with_content(init_file, None)

  # Copy .py files.
  files = BASE_DIR.rglob(r'*')
  extentions = {'.py', '.txt', '.md'}

  relative_pys = []
  init_pys = []
  for path in files:
    name = str(path)
    if path.is_dir():
      continue
    if path.suffix not in extentions:
      continue
    if ('pip_package' in name) or ('_test.py' in name):
      continue

    build_path = lib_pkg.joinpath(path.relative_to(BASE_DIR))
    _ensure_dir_created(build_path.parent)
    shutil.copy2(str(path), str(build_path))

    if path.suffix == '.py':
      if path.stem != '__init__':
        relative_pys.append(build_path.relative_to(lib_pkg))
      else:
        init_pys.append(build_path.relative_to(lib_pkg))

  # Create namespace mapping.
  offical_names = OFFICIAL_NAMESPACE.split('.')
  official_pkg = build_root.joinpath(*offical_names)

  # Prepare __init__.py.
  _ensure_dir_created(official_pkg)
  for i in range(len(offical_names) + 1):
    dirpath = build_root.joinpath(*offical_names[:i])
    init_file = dirpath.joinpath('__init__.py')
    if not init_file.exists():
      _create_py_with_content(init_file, None)

  # Import lib paths by adding like:
  # from tensorflow_examples.lite.model_maker.core.task.text_classifier import *
  for r in relative_pys:
    official_py = official_pkg.joinpath(r)

    ns = str(r).replace('.py', '').replace(os.sep, '.')
    ns = '{}.{}'.format(LIB_NAMESPACE, ns)
    content = 'from {} import *'.format(ns)
    _ensure_dir_created(official_py.parent)
    _create_py_with_content(official_py, content)

  # Copy the __init__.py.
  for p in init_pys:
    build_py = lib_pkg.joinpath(p)
    official_py = official_pkg.joinpath(p)
    shutil.copy2(str(build_py), str(official_py))

  # Return package.
  namespace_packages = find_namespace_packages(where=build_root)
  package_dir_mapping = {'': SRC_NAME}
  return namespace_packages, package_dir_mapping


packages, package_dir = prepare_package_src()

setup(
    name=project_name,
    version=version,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Google LLC',
    author_email='packages@tensorflow.org',
    url='http://github.com/tensorflow/examples',
    download_url='https://github.com/tensorflow/examples/tags',
    license='Apache 2.0',
    package_dir=package_dir,
    packages=packages,
    scripts=[],
    install_requires=get_required_packages(),
    entry_points={
        'console_scripts': [MODEL_MAKER_CONSOLE,],
    },
    classifiers=classifiers,
    keywords=['tensorflow', 'lite', 'model customization', 'transfer learning'],
)
