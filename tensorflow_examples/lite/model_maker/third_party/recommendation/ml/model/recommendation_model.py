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
"""On-device personalized recommendation model."""
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.configs import input_config_pb2
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.configs import model_config as model_config_class
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import context_encoder
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import dotproduct_similarity
from tensorflow_examples.lite.model_maker.third_party.recommendation.ml.model import label_encoder


class RecommendationModel(tf.keras.Model):
  """Personalized dual-encoder style recommendation model."""

  def __init__(self,
               input_config: input_config_pb2.InputConfig,
               model_config: model_config_class.ModelConfig):
    """Initializes RecommendationModel according to input and model configs.

    Takes in input and model configs to initialize the recommendation model.
    Context encoder layer, label encoder layer and dotproduct similarity layer
    will be prepared.

    Args:
      input_config: The input config (input_config_pb2.InputConfig).
      model_config: The model config (model_config_class.ModelConfig).
    """
    super(RecommendationModel, self).__init__()
    self._input_config = input_config
    self._model_config = model_config
    self._context_encoder = context_encoder.ContextEncoder(
        input_config=self._input_config, model_config=self._model_config)
    self._label_encoder = label_encoder.LabelEncoder(
        input_config=self._input_config)
    self._dotproduct_layer = dotproduct_similarity.DotProductSimilarity()

  def call(self, inputs):
    """Compute outputs by passing inputs through the model.

    Computes the dotproduct similarity between the context embeddings with
    label embeddings of all items in the vocabulary, hence the output could be
    used to compute loss with all non-label items as negatives.

    Args:
      inputs: The inputs to the model, which should be a dictionary having
        feature names as keys and parsed inputs generated with input pipeline.

    Returns:
      Dotproduct similarity for training mode.
    """
    context_embeddings = self._context_encoder(inputs)
    # Compute the similarities between the context embedding and embeddings of
    # all items in the vocabulary.
    full_vocab_input_label = tf.range(
        self._input_config.label_feature.vocab_size)
    label = {self._label_encoder.label_name: full_vocab_input_label}
    full_vocab_label_embeddings = self._label_encoder(label)
    full_vocab_dotproduct = self._dotproduct_layer(
        context_embeddings=context_embeddings,
        label_embeddings=full_vocab_label_embeddings,
        top_k=None)[0]
    return full_vocab_dotproduct

  @tf.function
  def serve(self, **kwargs):
    inputs = kwargs
    context_embeddings = self._context_encoder(inputs)
    full_vocab_input_label = tf.range(
        self._input_config.label_feature.vocab_size)
    full_vocab_label_embeddings = self._label_encoder.encode(
        full_vocab_input_label)
    assert self._model_config.num_predictions
    (_, top_ids, top_scores) = self._dotproduct_layer(
        context_embeddings=context_embeddings,
        label_embeddings=full_vocab_label_embeddings,
        top_k=self._model_config.num_predictions)
    return {'top_prediction_ids': top_ids, 'top_prediction_scores': top_scores}
