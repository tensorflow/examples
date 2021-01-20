# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for recommendation task."""
import os

import tensorflow.compat.v2 as tf

from tensorflow_examples.lite.model_maker.core.data_util import recommendation_dataloader as _dl
from tensorflow_examples.lite.model_maker.core.data_util import recommendation_testutil as _testutil
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import recommendation


class RecommendationTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    _testutil.setup_fake_testdata(self)
    with _testutil.patch_download_and_extract_data(self.movielens_dir):
      self.train_loader = _dl.RecommendationDataLoader.from_movielens(
          self.generated_dir, 'train', self.test_tempdir)
      self.test_loader = _dl.RecommendationDataLoader.from_movielens(
          self.generated_dir, 'test', self.test_tempdir)

    self.model_spec_options = dict(
        context_embedding_dim=16,
        label_embedding_dim=16,
        item_vocab_size=self.test_loader.max_vocab_id,
        hidden_layer_dim_ratios=[1, 1],
    )

  def test_create(self):
    model_dir = os.path.join(self.test_tempdir, 'recommendation_create')
    model = recommendation.create(
        self.train_loader,
        'recommendation_bow',
        self.model_spec_options,
        model_dir,
        steps_per_epoch=1)
    self.assertIsNotNone(model.model)

  def test_evaluate(self):
    model_dir = os.path.join(self.test_tempdir, 'recommendation_evaluate')
    model = recommendation.create(
        self.train_loader,
        'recommendation_bow',
        self.model_spec_options,
        model_dir,
        steps_per_epoch=1)
    history = model.evaluate(self.test_loader)
    self.assertIsInstance(history, list)
    self.assertTrue(history)  # Non-empty list.

  def test_export(self):
    model_dir = os.path.join(self.test_tempdir, 'recommendation_export')
    model = recommendation.create(
        self.train_loader,
        'recommendation_bow',
        self.model_spec_options,
        model_dir,
        steps_per_epoch=1)
    export_format = [ExportFormat.TFLITE, ExportFormat.SAVED_MODEL]
    model.export(model_dir, export_format=export_format)
    # Expect tflite file.
    expected_tflite = os.path.join(model_dir, 'model.tflite')
    self.assertTrue(os.path.exists(expected_tflite))
    self.assertGreater(os.path.getsize(expected_tflite), 0)

    # Expect saved model.
    expected_saved_model = os.path.join(model_dir, 'saved_model',
                                        'saved_model.pb')
    self.assertTrue(os.path.exists(expected_saved_model))
    self.assertGreater(os.path.getsize(expected_saved_model), 0)

    # Evaluate tflite model.
    self._test_evaluate_tflite(model, expected_tflite)

  def _test_evaluate_tflite(self, model, tflite_filepath):
    result = model.evaluate_tflite(tflite_filepath, self.test_loader)
    self.assertIsInstance(result, dict)
    self.assertTrue(result)  # Not empty.


if __name__ == '__main__':
  tf.test.main()
