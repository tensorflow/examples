# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for searcher."""
import os
import tempfile

from absl.testing import parameterized
import numpy as np
import requests
import tensorflow as tf
from tensorflow_lite_support.metadata.python import metadata as _metadata
from tensorflow_examples.lite.model_maker.core.data_util import image_searcher_dataloader
from tensorflow_examples.lite.model_maker.core.data_util import searcher_dataloader
from tensorflow_examples.lite.model_maker.core.task import searcher
from tensorflow_lite_support.python.task.text import text_searcher as task_text_searcher
from tensorflow_lite_support.python.task.vision import image_searcher as task_image_searcher
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_examples.lite.model_maker.core import test_util


class SearcherTest(tf.test.TestCase, parameterized.TestCase):

  def test_searcher_with_image_embedder(self):
    tflite_path = test_util.get_test_data_path(
        "mobilenet_v2_035_96_embedder_with_metadata.tflite")
    image_folder = test_util.get_test_data_path("animals")

    # Creates the image searcher data loader.
    data_loader = image_searcher_dataloader.DataLoader.create(tflite_path)
    data_loader.load_from_folder(image_folder)
    # Expands the data by 50x.
    repeated_times = 500
    data_loader._dataset = np.repeat(
        data_loader.dataset, repeated_times, axis=0)
    data_loader._metadata = repeated_times * data_loader.metadata

    # Creates searcher model with ScaNN options including `score_ah`.
    scann_options = searcher.ScaNNOptions(
        distance_measure="dot_product",
        tree=searcher.Tree(num_leaves=10, num_leaves_to_search=2),
        score_ah=searcher.ScoreAH(2, anisotropic_quantization_threshold=0.2))
    model = searcher.Searcher.create_from_data(data_loader, scann_options)

    # Exports the model to on-device ScaNN index file.
    with tempfile.TemporaryDirectory() as tmpdir:
      # Exports the standalone on-device ScaNN index file.
      output_scann_path = os.path.join(tmpdir, "on_device_scann_index.ldb")
      model.export(
          export_filename=output_scann_path,
          userinfo=b"userinfo\x00\x11\x22\x34",
          export_format=searcher.ExportFormat.SCANN_INDEX_FILE)

      self.assertTrue(os.path.exists(output_scann_path))
      self.assertGreater(os.path.getsize(output_scann_path), 0)

      test_image = tensor_image.TensorImage.create_from_file(
          test_util.get_test_data_path("sparrow.png"))
      searcher_infer = task_image_searcher.ImageSearcher.create_from_file(
          tflite_path, output_scann_path)

      result1 = searcher_infer.search(test_image)
      # Don't check the expected result directly since the result changes with
      # `score_ah` options.
      self.assertLen(result1.nearest_neighbors, 5)

      # Exports the TFLite with on-device ScaNN index file as the associate
      # file.
      output_tflite_path = os.path.join(tmpdir, "model.tflite")
      model.export(
          export_filename=output_tflite_path,
          userinfo=b"userinfo\x00\x11\x22\x34",
          export_format=searcher.ExportFormat.TFLITE)

      self.assertTrue(os.path.exists(output_tflite_path))
      self.assertGreater(os.path.getsize(output_tflite_path), 0)

      displayer = _metadata.MetadataDisplayer.with_model_file(
          output_tflite_path)
      actual_json = displayer.get_metadata_json()
      expected_json_file = test_util.get_test_data_path(
          "mobilenet_v3_small_100_224_embedder_scann.json")
      with open(expected_json_file, "r") as f:
        expected_json = f.read()
      self.assertEqual(actual_json, expected_json)

      searcher_infer = task_image_searcher.ImageSearcher.create_from_file(
          output_tflite_path)
      result2 = searcher_infer.search(test_image)
      self.assertLen(result2.nearest_neighbors, 5)

  def test_searcher_with_text_embedder(self):
    # Dumpy text embedder TFLite model with string as input which doesn't need
    # additional preprocessing in the task library.
    tflite_path = test_util.get_test_data_path("dummy_sp_text_embedder.tflite")

    np.random.seed(123)
    tf.random.set_seed(123)

    # Creates the base SearcherDataLoader initialized from tflite file, dataset
    # and metadata.
    dataset_size = 1000
    dim = 8
    data_loader = searcher_dataloader.DataLoader(
        embedder_path=tflite_path,
        dataset=np.random.rand(dataset_size, dim),
        metadata=["0"] * dataset_size)

    # Creates searcher model with ScaNN options including `score_brute_force`.
    scann_options = searcher.ScaNNOptions(
        distance_measure="dot_product",
        tree=searcher.Tree(num_leaves=10, num_leaves_to_search=2),
        score_brute_force=searcher.ScoreBruteForce())
    model = searcher.Searcher.create_from_data(data_loader, scann_options)

    with tempfile.TemporaryDirectory() as tmpdir:
      # Exports the standalone on-device ScaNN index file.
      output_scann_path = os.path.join(tmpdir, "text_on_device_scann_index.ldb")
      model.export(
          export_filename=output_scann_path,
          userinfo=b"userinfo\x00\x11\x22\x34",
          export_format=searcher.ExportFormat.SCANN_INDEX_FILE)

      self.assertTrue(os.path.exists(output_scann_path))
      self.assertGreater(os.path.getsize(output_scann_path), 0)

      searcher_infer = task_text_searcher.TextSearcher.create_from_file(
          tflite_path, output_scann_path)
      test_text = "Test text."
      result = searcher_infer.search(test_text)
      # TODO(b/231393039): Changes to check the expected result if fix the
      # inconsistency issue between the internal and external result.
      self.assertLen(result.nearest_neighbors, 5)

      # Exports the TFLite with on-device ScaNN index file as the associate
      # file.
      output_tflite_path = os.path.join(tmpdir, "text_model.tflite")
      model.export(
          export_filename=output_tflite_path,
          userinfo="",
          export_format=searcher.ExportFormat.TFLITE)

      self.assertTrue(os.path.exists(output_tflite_path))
      self.assertGreater(os.path.getsize(output_tflite_path), 0)

      searcher_infer = task_text_searcher.TextSearcher.create_from_file(
          output_tflite_path)
      result = searcher_infer.search(test_text)
      self.assertLen(result.nearest_neighbors, 5)

  @parameterized.parameters(
      ("universal_sentence_encoder_embedder.tflite",
       "https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1?lite-format=tflite",
       100),
      ("mobilebert_embedding_with_metadata.tflite",
       " https://storage.googleapis.com/download.tensorflow.org/models/tflite_support/bert_embedder/mobilebert_embedding_with_metadata.tflite",
       512),
  )
  def test_searcher_with_3_inputs_models(self, tflite_filename, url, dim):
    # Tests seacher with the universal sentence encoder model or bert model.
    # Gets the path to the tflite embedder model.
    try:
      tflite_path = test_util.get_test_data_path(tflite_filename)
    except ValueError:
      # Used for external test, download the tflite model in the tensorflow
      # hub.
      r = requests.get(url)
      tflite_path = os.path.join(self.get_temp_dir(), "embedder.tflite")
      with open(tflite_path, "wb") as f:
        f.write(r.content)

    # Gets the data loader.
    data_loader = searcher_dataloader.DataLoader(
        embedder_path=tflite_path,
        dataset=np.random.rand(200, dim),
        metadata=["1"] * 200)

    # Creates searcher model with ScaNN options.
    scann_options = searcher.ScaNNOptions(
        distance_measure="dot_product",
        tree=searcher.Tree(num_leaves=10, num_leaves_to_search=2),
        score_brute_force=searcher.ScoreBruteForce())
    model = searcher.Searcher.create_from_data(data_loader, scann_options)

    # Exports the standalone on-device ScaNN index file.
    output_scann_path = os.path.join(self.get_temp_dir(), "scann_index.ldb")
    model.export(
        export_filename=output_scann_path,
        userinfo="",
        export_format=searcher.ExportFormat.SCANN_INDEX_FILE)

    self.assertTrue(os.path.exists(output_scann_path))
    self.assertGreater(os.path.getsize(output_scann_path), 0)

    # Runs the inference to see if it works.
    searcher_infer = task_text_searcher.TextSearcher.create_from_file(
        tflite_path, output_scann_path)
    result = searcher_infer.search("The weather is good.")
    self.assertLen(result.nearest_neighbors, 5)

    # Exports the TFLite with on-device ScaNN index file as the associate
    # file.
    output_tflite_path = os.path.join(self.get_temp_dir(), "searcher.tflite")
    model.export(
        export_filename=output_tflite_path,
        userinfo="",
        export_format=searcher.ExportFormat.TFLITE)

    self.assertTrue(os.path.exists(output_tflite_path))
    self.assertGreater(os.path.getsize(output_tflite_path), 0)

    # Runs the inference to see if it works.
    searcher_infer = task_text_searcher.TextSearcher.create_from_file(
        output_tflite_path)
    result = searcher_infer.search("The weather is so bad.")
    self.assertLen(result.nearest_neighbors, 5)

if __name__ == "__main__":
  tf.test.main()
