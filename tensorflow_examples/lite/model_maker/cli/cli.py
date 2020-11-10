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
from tensorflow_examples.lite.model_maker.demo import question_answer_demo
from tensorflow_examples.lite.model_maker.demo import text_classification_demo

_IMAGE_MODELS = model_spec.IMAGE_CLASSIFICATION_MODELS
_TEXT_MODELS = model_spec.TEXT_CLASSIFICATION_MODELS
_QA_MODELS = model_spec.QUESTION_ANSWERING_MODELS


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
                           export_dir,
                           spec='efficientnet_lite0',
                           **kwargs):
    """Run Image classification.

    Args:
      data_dir: str, input directory of training data. (required)
      export_dir: str, output directory to export files. (required)
      spec: str, model_name. Valid: {MODELS}, default: efficientnet_lite0.
      **kwargs: --epochs: int, epoch num to run. More: see `create` function.
    """
    # Convert types
    data_dir = str(data_dir)
    export_dir = str(export_dir)

    image_classification_demo.run(data_dir, export_dir, spec, **kwargs)

  @FormatDoc(MODELS=_TEXT_MODELS)
  def text_classification(self,
                          data_dir,
                          export_dir,
                          spec='mobilebert_classifier',
                          **kwargs):
    r"""Run text classification.

    Args:
      data_dir: str, input directory of training data. (required)
      export_dir: str, output directory to export files. (required)
      spec: str, model_name. Valid: {MODELS}, default: mobilebert_classifier.
      **kwargs: --epochs: int, epoch num to run. More: see `create` function.
    """
    # Convert types
    data_dir = str(data_dir)
    export_dir = str(export_dir)
    text_classification_demo.run(data_dir, export_dir, spec, **kwargs)

  @FormatDoc(MODELS=_QA_MODELS)
  def question_answer(self,
                      train_data_path,
                      validation_data_path,
                      export_dir,
                      spec='mobilebert_qa_squad',
                      **kwargs):
    r"""Run question answer.

    Args:
      train_data_path: str, input path of training data. (required)
      validation_data_path: str, input path of training data. (required)
      export_dir: str, output directory to export files. (required)
      spec: str, model_name. Valid: {MODELS}, default: mobilebert_qa.
      **kwargs: --epochs: int, epoch num to run. More: see `create` function.
    """
    # Convert types
    train_data_path = str(train_data_path)
    validation_data_path = str(validation_data_path)
    question_answer_demo.run(train_data_path, validation_data_path, export_dir,
                             spec, **kwargs)


def main():
  fire.Fire(ModelMakerCLI)


if __name__ == '__main__':
  main()
