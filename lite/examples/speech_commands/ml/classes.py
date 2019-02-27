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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from generator import prepare_words_list


def get_classes(wanted_only=False):
  if wanted_only:
    classes = 'stop down off right up go on yes left no'
    classes = classes.split(' ')
    assert len(classes) == 10
  else:
    classes = ('sheila nine stop bed four six down bird marvin cat off right '
               'seven eight up three happy go zero on wow dog yes five one tree'
               ' house two left no')  # noqa
    classes = classes.split(' ')
    assert len(classes) == 30
  return classes


def get_int2label(wanted_only=False, extend_reversed=False):
  classes = get_classes(
      wanted_only=wanted_only, extend_reversed=extend_reversed)
  classes = prepare_words_list(classes)
  int2label = {i: l for i, l in enumerate(classes)}
  int2label = OrderedDict(sorted(int2label.items(), key=lambda x: x[0]))
  return int2label


def get_label2int(wanted_only=False, extend_reversed=False):
  classes = get_classes(
      wanted_only=wanted_only, extend_reversed=extend_reversed)
  classes = prepare_words_list(classes)
  label2int = {l: i for i, l in enumerate(classes)}
  label2int = OrderedDict(sorted(label2int.items(), key=lambda x: x[1]))
  return label2int
