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
"""CLI tool for model maker."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fire

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.task import model_spec
from tensorflow_examples.lite.model_maker.demo import image_classification_demo
from tensorflow_examples.lite.model_maker.demo import text_classification_demo

_IMAGE_MODELS = model_spec.IMAGE_CLASSIFICATION_MODELS
_TEXT_MODELS = model_spec.TEXT_CLASSIFICATION_MODELS


class FormatDoc(object):
  """Decorator to format functionn doc with given parameters."""

  def __init__(self, *format_args, **format_kwargs):
    self.args = format_args
    self.kwargs = format_kwargs

  def __call__(self, fn):
    fn.__doc__ = fn.__doc__.format(*self.args, **self.kwargs)
    return fn


class ModelMakerCLI(object):
  """Model Maker Command Line Interface.

  Flags:
    --tf: int, version of TF behavior. Valid: [1, 2], default: 2.
  """

  def __init__(self, tf=2):
    compat.setup_tf_behavior(tf)

  @FormatDoc(MODELS=_IMAGE_MODELS)
  def image_classification(self,
                           data_dir,
                           tflite_filename,
                           label_filename,
                           spec='efficientnet_b0',
                           **kwargs):
    """Run Image classification.

    Args:
      data_dir: str, input directory of training data. (required)
      tflite_filename: str, output path to export tflite file. (required)
      label_filename: str, output path to export label file. (required)
      spec: str, model_name. Valid: {MODELS}, default: efficientnet_b0.
      **kwargs: --epochs: int, epoch num to run. More: see `create` function.
    """
    # Convert types
    data_dir = str(data_dir)
    tflite_filename = str(tflite_filename)
    label_filename = str(label_filename)

    image_classification_demo.run(data_dir, tflite_filename, label_filename,
                                  spec, **kwargs)

  @FormatDoc(MODELS=_TEXT_MODELS)
  def text_classification(self,
                          data_dir,
                          tflite_filename,
                          label_filename,
                          vocab_filename,
                          spec='bert',
                          **kwargs):
    r"""Run text classification.

    Args:
      data_dir: str, input directory of training data. (required)
      tflite_filename: str, output path to export tflite file. (required)
      label_filename: str, output path to export label file. (required)
      vocab_filename: str, output path to export vocab file. (required)
      spec: str, model_name. Valid: {MODELS}, default: bert.
      **kwargs: --epochs: int, epoch num to run. More: see `create` function.
    """
    # Convert types
    data_dir = str(data_dir)
    tflite_filename = str(tflite_filename)
    label_filename = str(label_filename)
    vocab_filename = str(vocab_filename)

    text_classification_demo.run(data_dir, tflite_filename, label_filename,
                                 vocab_filename, spec, **kwargs)


def main():
  fire.Fire(ModelMakerCLI)


if __name__ == '__main__':
  main()
