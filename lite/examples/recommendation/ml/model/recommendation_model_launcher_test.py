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
"""Tests for recommendation_model_launcher."""
import os
from absl import flags
import tensorflow as tf
from model import input_pipeline
from model import recommendation_model_launcher as launcher
from google.protobuf import text_format

FLAGS = flags.FLAGS

FAKE_MOVIE_GENRE_VOCAB = [
    'UNK',
    'Comedy',
    'Drama',
    'Romance',
    'Animation',
    'Children'
]

TEST_INPUT_CONFIG = """
    activity_feature_groups {
      features {
        feature_name: "context_movie_id"
        feature_type: INT
        vocab_size: 3952
        embedding_dim: 8
        feature_length: 5
      }
      features {
        feature_name: "context_movie_rating"
        feature_type: FLOAT
        feature_length: 5
      }
      encoder_type: CNN
    }
    activity_feature_groups {
      features {
        feature_name: "context_movie_genre"
        feature_type: STRING
        vocab_name: "movie_genre_vocab.txt"
        vocab_size: 19
        embedding_dim: 8
        feature_length: 8
      }
      encoder_type: BOW
    }
    label_feature {
      feature_name: "label_movie_id"
      feature_type: INT
      vocab_size: 3952
      embedding_dim: 8
      feature_length: 1
    }
"""

EXAMPLE1 = text_format.Parse(
    """
    features {
        feature {
          key: "context_movie_id"
          value {
            int64_list {
              value: [1, 2, 0, 0, 0]
            }
          }
        }
        feature {
          key: "context_movie_rating"
          value {
            float_list {
              value: [3.5, 4.0, 0.0, 0.0, 0.0]
            }
          }
        }
        feature {
          key: "context_movie_genre"
          value {
            bytes_list {
              value: [
                    "Animation", "Children", "Comedy", "Comedy", "Romance", "UNK", "UNK", "UNK"
                ]
            }
          }
        }
        feature {
          key: "label_movie_id"
          value {
            int64_list {
              value: [3]
            }
          }
        }
      }""", tf.train.Example())


class RecommendationModelLauncherTest(tf.test.TestCase):

  def _AssertSparseTensorValueEqual(self, a, b):
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def _assertInputDetail(self, input_details, index, name, shape):
    self.assertEqual(name, input_details[index]['name'])
    self.assertEqual(shape, input_details[index]['shape'])

  def setUp(self):
    super().setUp()
    self.tmp_dir = self.create_tempdir()
    self.test_input_config_file = os.path.join(self.tmp_dir,
                                               'input_config.pbtxt')
    self.test_movie_genre_vocab_file = os.path.join(self.tmp_dir,
                                                    'movie_genre_vocab.txt')
    self.test_input_data_file = os.path.join(self.tmp_dir,
                                             'test_input_data.tfrecord')
    with open(self.test_input_config_file, 'w', encoding='utf-8') as f:
      f.write(TEST_INPUT_CONFIG)
    with open(self.test_movie_genre_vocab_file, 'w', encoding='utf-8') as f:
      for item in FAKE_MOVIE_GENRE_VOCAB:
        f.write(item + '\n')
    with tf.io.TFRecordWriter(self.test_input_data_file) as file_writer:
      file_writer.write(EXAMPLE1.SerializeToString())

    self.test_model_dir = os.path.join(self.tmp_dir, 'test_model_dir')

    FLAGS.training_data_filepattern = self.test_input_data_file
    FLAGS.testing_data_filepattern = self.test_input_data_file
    FLAGS.input_config_file = self.test_input_config_file
    FLAGS.model_dir = self.test_model_dir
    FLAGS.hidden_layer_dims = [8, 4]
    FLAGS.eval_top_k = [1, 5]
    FLAGS.num_predictions = 5
    FLAGS.conv_num_filter_ratios = [2, 4]
    FLAGS.conv_kernel_size = 4
    FLAGS.lstm_num_units = 16

  def testModelTrainEvalExport(self):
    """Verifies that model can be trained and evaluated."""
    tf.io.gfile.mkdir(FLAGS.model_dir)
    input_config = launcher.load_input_config()
    model_config = launcher.prepare_model_config()
    dataset = input_pipeline.get_input_dataset(
        data_filepattern=self.test_input_data_file,
        input_config=input_config,
        vocab_file_dir=self.tmp_dir,
        batch_size=8)
    model = launcher.build_keras_model(input_config, model_config)
    launcher.train_and_eval(
        model=model,
        model_dir=FLAGS.model_dir,
        train_input_dataset=dataset,
        eval_input_dataset=dataset,
        steps_per_epoch=2,
        epochs=2,
        eval_steps=1)
    self.assertTrue(os.path.exists(self.test_model_dir))
    summaries_dir = os.path.join(self.test_model_dir, 'summaries')
    self.assertTrue(os.path.exists(summaries_dir))
    export_dir = os.path.join(FLAGS.model_dir, 'export')
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
    launcher.save_model(
        checkpoint_path=latest_checkpoint,
        export_dir=export_dir,
        input_config=input_config,
        model_config=model_config)
    savedmodel_path = os.path.join(export_dir, 'saved_model.pb')
    self.assertTrue(os.path.exists(savedmodel_path))
    imported = tf.saved_model.load(export_dir, tags=None)
    infer = imported.signatures['serving_default']
    context_movie_id = tf.range(5, dtype=tf.int32)
    context_movie_rating = tf.range(5, dtype=tf.float32)
    context_movie_genre = tf.range(8, dtype=tf.int32)
    predictions = infer(context_movie_id=context_movie_id,
                        context_movie_rating=context_movie_rating,
                        context_movie_genre=context_movie_genre)
    self.assertAllEqual([5], predictions['top_prediction_ids'].shape)
    self.assertAllEqual([5], predictions['top_prediction_scores'].shape)
    launcher.export_tflite(export_dir)
    tflite_model_path = os.path.join(export_dir, 'model.tflite')
    self.assertTrue(os.path.exists(tflite_model_path))
    f = open(tflite_model_path, 'rb')
    interpreter = tf.lite.Interpreter(model_content=f.read())
    interpreter.allocate_tensors()
    inference_signature = interpreter.get_signature_list()['serving_default']
    self.assertAllEqual(
        ['context_movie_genre', 'context_movie_id', 'context_movie_rating'],
        inference_signature['inputs'])
    self.assertAllEqual(['top_prediction_ids', 'top_prediction_scores'],
                        inference_signature['outputs'])
    serving_name_to_tenors = {
        'serving_default_context_movie_id:0': context_movie_id,
        'serving_default_context_movie_rating:0': context_movie_rating,
        'serving_default_context_movie_genre:0': context_movie_genre
    }
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    indice_to_tensors = {}
    for input_detail in input_details:
      indice_to_tensors[input_detail['index']] = serving_name_to_tenors[
          input_detail['name']]
    for index, tensor in indice_to_tensors.items():
      interpreter.set_tensor(index, tensor)
    interpreter.invoke()
    tflite_top_predictions_ids = interpreter.get_tensor(
        output_details[0]['index'])
    tflite_top_prediction_scores = interpreter.get_tensor(
        output_details[1]['index'])
    self.assertAllEqual([5], tflite_top_predictions_ids.shape)
    self.assertAllEqual([5], tflite_top_prediction_scores.shape)

if __name__ == '__main__':
  tf.test.main()
