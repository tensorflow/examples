# Lint as: python3
#   Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for recommendation_model_launcher_keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import flags
import tensorflow as tf
from model import recommendation_model_launcher_keras as launcher

FLAGS = flags.FLAGS


def _int64_feature(value_list):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


EXAMPLE_1 = tf.train.Example(
    features=tf.train.Features(
        feature={
            'context': _int64_feature([3185, 1269, 3170, 0, 0]),
            'label': _int64_feature([1021])
        })).SerializeToString()


class RecommendationModelLauncherTest(tf.test.TestCase):

  def setUp(self):
    super(RecommendationModelLauncherTest, self).setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.train_tfrecord_file = os.path.join(self.tmp_dir, 'train.tfrecord')
    self.test_tfrecord_file = os.path.join(self.tmp_dir, 'test.tfrecord')
    self.test_model_dir = os.path.join(self.tmp_dir, 'test_model_dir')
    self.params = {
        'context_embedding_dim': 16,
        'label_embedding_dim': 16,
        'hidden_layer_dim_ratios': [1, 1],
        'item_vocab_size': 3952,
        'eval_top_k': [1],
        'conv_num_filter_ratios': [2, 4],
        'conv_kernel_size': 4,
        'lstm_num_units': 16
    }

    with tf.io.TFRecordWriter(
        self.train_tfrecord_file, options=tf.io.TFRecordOptions()) as writer:
      writer.write(EXAMPLE_1)
    with tf.io.TFRecordWriter(
        self.test_tfrecord_file, options=tf.io.TFRecordOptions()) as writer:
      writer.write(EXAMPLE_1)

    FLAGS.training_data_filepattern = self.train_tfrecord_file
    FLAGS.testing_data_filepattern = self.test_tfrecord_file
    FLAGS.model_dir = self.test_model_dir
    FLAGS.encoder_type = 'cnn'
    FLAGS.num_predictions = 10
    FLAGS.max_history_length = 10
    FLAGS.batch_size = 1

  def testModelFnTrainModeExecute(self):
    """Verifies that 'model_fn' can be executed in train and eval mode."""
    self.params['encoder_type'] = FLAGS.encoder_type
    train_input_fn = launcher.InputFn(FLAGS.training_data_filepattern,
                                      FLAGS.batch_size)
    eval_input_fn = launcher.InputFn(FLAGS.testing_data_filepattern,
                                     FLAGS.batch_size)
    model = launcher.build_keras_model(self.params, FLAGS.learning_rate,
                                       FLAGS.gradient_clip_norm)
    launcher.train_and_eval(
        model=model,
        model_dir=FLAGS.model_dir,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        steps_per_epoch=2,
        epochs=2,
        eval_steps=1)
    self.assertTrue(os.path.exists(self.test_model_dir))
    summaries_dir = os.path.join(self.test_model_dir, 'summaries')
    self.assertTrue(os.path.exists(summaries_dir))

  def testModelFnExportModeExecute(self):
    """Verifies model can be exported to savedmodel and tflite model."""
    self.params['encoder_type'] = FLAGS.encoder_type
    self.params['num_predictions'] = FLAGS.num_predictions
    train_input_fn = launcher.InputFn(FLAGS.training_data_filepattern,
                                      FLAGS.batch_size)
    eval_input_fn = launcher.InputFn(FLAGS.testing_data_filepattern,
                                     FLAGS.batch_size)
    model = launcher.build_keras_model(self.params, FLAGS.learning_rate,
                                       FLAGS.gradient_clip_norm)
    launcher.train_and_eval(
        model=model,
        model_dir=FLAGS.model_dir,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        steps_per_epoch=2,
        epochs=2,
        eval_steps=1)
    export_dir = os.path.join(FLAGS.model_dir, 'export')
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
    launcher.export(
        checkpoint_path=latest_checkpoint,
        export_dir=export_dir,
        params=self.params,
        max_history_length=FLAGS.max_history_length)
    savedmodel_path = os.path.join(export_dir, 'saved_model.pb')
    self.assertTrue(os.path.exists(savedmodel_path))
    imported = tf.saved_model.load(export_dir, tags=None)
    infer = imported.signatures['serving_default']
    context = tf.range(10)
    predictions = infer(context)
    self.assertAllEqual([10], predictions['top_prediction_ids'].shape)
    self.assertAllEqual([10], predictions['top_prediction_scores'].shape)
    launcher.export_tflite(export_dir)
    tflite_model_path = os.path.join(export_dir, 'model.tflite')
    self.assertTrue(os.path.exists(tflite_model_path))
    f = open(tflite_model_path, 'rb')
    interpreter = tf.lite.Interpreter(model_content=f.read())
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    self.assertEqual([10], input_details[0]['shape'])
    self.assertEqual('serving_default_context:0', input_details[0]['name'])
    interpreter.set_tensor(input_details[0]['index'], context)
    interpreter.invoke()
    tflite_top_predictions_ids = interpreter.get_tensor(
        output_details[0]['index'])
    tflite_top_prediction_scores = interpreter.get_tensor(
        output_details[1]['index'])
    self.assertAllEqual([10], tflite_top_predictions_ids.shape)
    self.assertAllEqual([10], tflite_top_prediction_scores.shape)


if __name__ == '__main__':
  launcher.define_flags()
  tf.test.main()
