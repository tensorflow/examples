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
"""Setuptools configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

setup(
    name='tfltransfer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.0.0rc0',
        'Pillow>=6.1.0,<7.0',
        'scipy>=1.3.0,<2.0',
    ],
    entry_points={
        'console_scripts': [
            'tflite-transfer-convert = tfltransfer.tflite_transfer_convert:main',
        ],
    },
)
