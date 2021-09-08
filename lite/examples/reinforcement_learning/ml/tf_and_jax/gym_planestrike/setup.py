# Lint as: python3
#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Setup file for the OpenAI gym environment."""
import os
import shutil

from setuptools import find_packages
from setuptools import setup

source = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../common.py'))
target = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'gym_planestrike/envs/common.py'))
shutil.copyfile(source, target)

setup(
    name='gym_planestrike',
    version='0.1',
    description='Board game Plane Strike',
    author='TensorFlow Authors',
    license='Apache License 2.0',
    packages=find_packages(),
    install_requires=['gym', 'numpy'])
