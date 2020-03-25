# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""tensorflow_examples is a package of tensorflow example code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup

nightly = False
if '--nightly' in sys.argv:
  nightly = True
  sys.argv.remove('--nightly')

project_name = 'tensorflow-examples'
# Get the current commit hash
version = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8')

if nightly:
  project_name = 'tensorflow-examples-nightly'
  datestring = datetime.datetime.now().strftime('%Y%m%d%H%M')
  version = '%s-dev%s' % (version, datestring)

DOCLINES = __doc__.split('\n')

REQUIRED_PKGS = [
    'absl-py',
    'six',
]

TESTS_REQUIRE = [
    'jupyter',
]

REQUIRMENTS = 'tensorflow_examples/lite/model_maker/requirements.txt'.replace('/', os.sep)
with open(REQUIRMENTS) as f:
  MODEL_MAKER_REQUIRE = [l.strip() for l in f.read().splitlines() if l.strip()]

METADATA_REQUIRE = [
    'tflite-support==0.1.0a0',
]
if sys.version_info.major == 3:
  # Packages only for Python 3
  pass
else:
  # Packages only for Python 2
  TESTS_REQUIRE.append('mock')
  REQUIRED_PKGS.append('futures')  # concurrent.futures

if sys.version_info < (3, 4):
  # enum introduced in Python 3.4
  REQUIRED_PKGS.append('enum34')

setup(
    name=project_name,
    version=version,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    url='http://github.com/tensorflow/examples',
    download_url='https://github.com/tensorflow/examples/tags',
    license='Apache 2.0',
    packages=find_packages(),
    scripts=[],
    install_requires=REQUIRED_PKGS,
    extras_require={
        'tests': TESTS_REQUIRE,
        'model_maker': MODEL_MAKER_REQUIRE,
        'metadata': METADATA_REQUIRE,
    },
    entry_points={
        'console_scripts': [
            'model_maker=tensorflow_examples.lite.model_maker.cli.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow examples',
)
