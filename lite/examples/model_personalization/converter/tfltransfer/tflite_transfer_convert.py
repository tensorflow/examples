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
"""CLI wrapper for tflite_transfer_converter.

Converts a pair of TF models to a TFLite transfer learning model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# pylint: disable=g-bad-import-order
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer import tflite_transfer_converter
# pylint: enable=g-bad-import-order


def main():
  parser = argparse.ArgumentParser(
      description='Combines two TF models into a transfer learning model')
  parser.add_argument(
      '--train_batch_size', help='Training batch size', type=int, default=20)
  parser.add_argument(
      '--num_classes',
      help='Number of classes for the output',
      type=int,
      default=4)

  # Base model configuration.
  base_group = parser.add_mutually_exclusive_group(required=True)
  base_group.add_argument(
      '--base_mobilenetv2',
      help='Use MobileNetV2 as the base model',
      dest='base_mobilenetv2',
      action='store_true')
  base_group.add_argument(
      '--base_model_dir',
      help='Use a SavedModel under a given path as the base model',
      type=str)
  parser.add_argument(
      '--base_quantize',
      help='Whether the base model should be quantized',
      dest='base_quantize',
      action='store_true')
  parser.set_defaults(base_quantize=False)

  # Head model configuration.
  head_group = parser.add_mutually_exclusive_group(required=True)
  head_group.add_argument(
      '--head_model_dir',
      help='Use a SavedModel under a given path as the head model',
      type=str)
  head_group.add_argument(
      '--head_softmax',
      help='Use SoftmaxClassifier for the head model',
      dest='head_softmax',
      action='store_true')
  parser.add_argument(
      '--head_l2_reg',
      help='L2 regularization parameter for SoftmaxClassifier',
      type=float)

  # Optimizer configuration.
  parser.add_argument(
      '--optimizer',
      required=True,
      type=str,
      choices=['sgd', 'adam'],
      help='Which optimizer should be used')
  parser.add_argument(
      '--sgd_learning_rate', help='Learning rate for SGD', type=float)

  parser.add_argument(
      '--out_model_dir',
      help='Where the generated transfer learning model is saved',
      required=True,
      type=str)
  args = parser.parse_args()

  if args.base_mobilenetv2:
    base = bases.MobileNetV2Base(quantize=args.base_quantize)
  else:
    base = bases.SavedModelBase(
        args.base_model_dir, quantize=args.base_quantize)

  if args.head_model_dir:
    head = heads.LogitsSavedModelHead(args.head_model_dir)
  else:
    head = heads.SoftmaxClassifierHead(
        args.train_batch_size,
        base.bottleneck_shape(),
        args.num_classes,
        l2_reg=args.head_l2_reg)

  if args.optimizer == 'sgd':
    if args.sgd_learning_rate is not None:
      optimizer = optimizers.SGD(args.sgd_learning_rate)
    else:
      raise RuntimeError(
          '--sgd_learning_rate is required when SGD is used as an optimizer')
  elif args.optimizer == 'adam':
    optimizer = optimizers.Adam()

  converter = tflite_transfer_converter.TFLiteTransferConverter(
      args.num_classes, base, head, optimizer, args.train_batch_size)
  converter.convert_and_save(args.out_model_dir)


if __name__ == '__main__':
  main()
