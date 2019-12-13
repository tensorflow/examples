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
"""Tests for tflite_transfer_converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import unittest

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# pylint: disable=g-bad-import-order
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer import tflite_transfer_converter
# pylint: enable=g-bad-import-order

DEFAULT_INPUT_SIZE = 64
DEFAULT_BATCH_SIZE = 128
LEARNING_RATE = 0.001


class TestTfliteTransferConverter(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TestTfliteTransferConverter, cls).setUpClass()
    cls._default_base_model_dir = tempfile.mkdtemp('tflite-transfer-test-base')
    model = tf.keras.Sequential([
        layers.Dense(
            units=DEFAULT_INPUT_SIZE, input_shape=(DEFAULT_INPUT_SIZE,))
    ])
    model.build()
    tf.keras.experimental.export_saved_model(model, cls._default_base_model_dir)

  def setUp(self):
    super(TestTfliteTransferConverter, self).setUp()
    self._default_base_model = bases.SavedModelBase(
        TestTfliteTransferConverter._default_base_model_dir)

  def test_mobilenet_v2_saved_model_and_keras_model(self):
    input_size = DEFAULT_INPUT_SIZE
    output_size = 5

    head_model = tf.keras.Sequential([
        layers.Dense(
            units=32,
            input_shape=(input_size,),
            activation='relu',
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01)),
        layers.Dense(
            units=output_size,
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01)),
    ])
    head_model.compile(loss='categorical_crossentropy', optimizer='sgd')

    converter = tflite_transfer_converter.TFLiteTransferConverter(
        output_size, self._default_base_model, heads.KerasModelHead(head_model),
        optimizers.SGD(LEARNING_RATE), DEFAULT_BATCH_SIZE)

    models = converter._convert()

    parameter_shapes = [(input_size, 32), (32,), (32, output_size),
                        (output_size,)]
    self.assertSignatureEqual(models['initialize'], [()], parameter_shapes)
    self.assertSignatureEqual(models['bottleneck'], [(1, input_size)],
                              [(1, input_size)])
    self.assertSignatureEqual(models['inference'],
                              [(1, input_size)] + parameter_shapes,
                              [(1, output_size)])
    self.assertSignatureEqual(models['optimizer'],
                              parameter_shapes + parameter_shapes,
                              parameter_shapes)

  def test_mobilenet_v2_saved_model_and_softmax_classifier_model(self):
    input_size = DEFAULT_INPUT_SIZE
    output_size = 5
    batch_size = DEFAULT_BATCH_SIZE

    converter = tflite_transfer_converter.TFLiteTransferConverter(
        output_size, self._default_base_model,
        heads.SoftmaxClassifierHead(batch_size, (input_size,), output_size),
        optimizers.SGD(LEARNING_RATE), batch_size)
    models = converter._convert()

    parameter_shapes = [(input_size, output_size), (output_size,)]
    self.assertSignatureEqual(models['initialize'], [()], parameter_shapes)
    self.assertSignatureEqual(models['bottleneck'], [(1, input_size)],
                              [(1, input_size)])
    self.assertSignatureEqual(models['train_head'],
                              [(batch_size, input_size),
                               (batch_size, output_size)] + parameter_shapes,
                              [()] + parameter_shapes)
    self.assertSignatureEqual(models['inference'],
                              [(1, input_size)] + parameter_shapes,
                              [(1, output_size)])
    self.assertSignatureEqual(models['optimizer'],
                              parameter_shapes + parameter_shapes,
                              parameter_shapes)

  def test_mobilenet_v2_base_and_softmax_classifier_model(self):
    input_size = 224
    output_size = 5
    batch_size = DEFAULT_BATCH_SIZE

    base = bases.MobileNetV2Base(image_size=input_size)
    head = heads.SoftmaxClassifierHead(batch_size, base.bottleneck_shape(),
                                       output_size)
    optimizer = optimizers.SGD(LEARNING_RATE)

    converter = tflite_transfer_converter.TFLiteTransferConverter(
        output_size, base, head, optimizer, batch_size)
    models = converter._convert()

    parameter_shapes = [(7 * 7 * 1280, output_size), (output_size,)]
    self.assertSignatureEqual(models['initialize'], [()], parameter_shapes)
    self.assertSignatureEqual(models['bottleneck'],
                              [(1, input_size, input_size, 3)],
                              [(1, 7, 7, 1280)])
    self.assertSignatureEqual(models['train_head'],
                              [(batch_size, 7, 7, 1280),
                               (batch_size, output_size)] + parameter_shapes,
                              [()] + parameter_shapes)
    self.assertSignatureEqual(models['inference'],
                              [(1, 7, 7, 1280)] + parameter_shapes,
                              [(1, output_size)])
    self.assertSignatureEqual(models['optimizer'],
                              parameter_shapes + parameter_shapes,
                              parameter_shapes)

  def test_mobilenet_v2_base_and_softmax_classifier_model_adam(self):
    input_size = 224
    output_size = 5
    batch_size = DEFAULT_BATCH_SIZE

    base = bases.MobileNetV2Base(image_size=input_size)
    head = heads.SoftmaxClassifierHead(batch_size, base.bottleneck_shape(),
                                       output_size)
    optimizer = optimizers.Adam()

    converter = tflite_transfer_converter.TFLiteTransferConverter(
        output_size, base, head, optimizer, batch_size)
    models = converter._convert()

    param_shapes = [(7 * 7 * 1280, output_size), (output_size,)]
    self.assertSignatureEqual(
        models['optimizer'],
        param_shapes + param_shapes + param_shapes + param_shapes + [()],
        param_shapes + param_shapes + param_shapes + [()])

  def assertSignatureEqual(self, model, expected_inputs, expected_outputs):
    interpreter = tf.lite.Interpreter(model_content=model)
    inputs = [
        input_['shape'].tolist() for input_ in interpreter.get_input_details()
    ]
    outputs = [
        output['shape'].tolist() for output in interpreter.get_output_details()
    ]
    self.assertEqual(inputs, [list(dims) for dims in expected_inputs])
    self.assertEqual(outputs, [list(dims) for dims in expected_outputs])


if __name__ == '__main__':
  unittest.main()
