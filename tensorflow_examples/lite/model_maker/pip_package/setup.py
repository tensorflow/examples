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
import sys
import setup_util

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

# Set package version.
if nightly:
  project_name = '{}-nightly'.format(project_name)
  version = '0.2.6'  # Version prefix, usually major.minor.micro.
  version = '{:s}.dev{:s}'.format(version, datestring)
  classifiers += [
      'Development Status :: 4 - Beta',
  ]
else:
  version = '0.2.5'

# Path to model_maker dir: <repo>/tensorflow_examples/lite/model_maker
BASE_DIR = pathlib.Path(os.path.abspath(__file__)).parents[1]
# Path to root dir: <repo>
ROOT_DIR = BASE_DIR.parents[2]
# Original namespace of the lib.
LIB_NAMESPACE = 'tensorflow_examples.lite.model_maker'
# Official package namespace for API. Used as code name.
API_NAMESPACE = 'tflite_model_maker'
# Internal package tflite_model_maker.python mapping internal packages.
INTERNAL_NAME = 'python'
MODEL_MAKER_CONSOLE = 'tflite_model_maker=tflite_model_maker.python.cli.cli:main'

# Build dir `pip_package/src`: copy all source code and create a package.
SRC_NAME = 'src'
BUILD_DIR = BASE_DIR.joinpath('pip_package').joinpath(SRC_NAME)

# Setup options.
setup_options = {
    'package_dir': {
        '': SRC_NAME
    },
    'entry_points': {
        'console_scripts': [MODEL_MAKER_CONSOLE,],
    },
}

DESCRIPTION = ('TFLite Model Maker: a model customization library for on-device'
               ' applications.')
with BASE_DIR.joinpath('README.md').open() as readme_file:
  LONG_DESCRIPTION = readme_file.read()


def _read_required_packages(fpath):
  with fpath.open() as f:
    required_pkgs = [l.strip() for l in f.read().splitlines()]
    required_pkgs = list(
        filter(lambda line: line and not line.startswith('#'), required_pkgs))
  return required_pkgs


def get_required_packages():
  """Gets packages inside requirements.txt."""
  # Gets model maker's required packages
  filename = 'requirements_nightly.txt' if nightly else 'requirements.txt'
  fpath = BASE_DIR.joinpath(filename)
  required_pkgs = _read_required_packages(fpath)

  return required_pkgs


extra_options = setup_util.PackageGen(BASE_DIR, ROOT_DIR, BUILD_DIR, nightly,
                                      version, LIB_NAMESPACE, API_NAMESPACE,
                                      INTERNAL_NAME).run()
setup_options.update(extra_options)

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
    scripts=[],
    install_requires=get_required_packages(),
    classifiers=classifiers,
    keywords=['tensorflow', 'lite', 'model customization', 'transfer learning'],
    **setup_options)
