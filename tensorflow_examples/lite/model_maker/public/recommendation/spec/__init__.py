# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-bad-import-order,redefined-builtin
"""APIs for recommendation specifications.

Example:
```python
input_spec = recommendation.spec.InputSpec(
    activity_feature_groups=[
        # Group #1: defines how features are grouped in the first Group.
        dict(
            features=[
                # First feature.
                dict(
                    feature_name='context_movie_id',  # Feature name
                    feature_type='INT',  # Feature type
                    vocab_size=3953,     # ID size (number of IDs)
                    embedding_dim=8,     # Projected feature embedding dim
                    feature_length=10,   # History length of 10.
                ),
                # Maybe more features...
            ],
            encoder_type='CNN',  # CNN encoder (e.g. CNN, LSTM, BOW)
        ),
        # Maybe more groups...
    ],
    label_feature=dict(
        feature_name='label_movie_id',  # Label feature name
        feature_type='INT',  # Label type
        vocab_size=3953,   # Label size (number of classes)
        embedding_dim=8,   # label embedding demension
        feature_length=1,  # Exactly 1 label
    ),
)

model_hparams = recommendation.spec.ModelHParams(
    hidden_layer_dims=[32, 32],  # Hidden layers dimension.
    eval_top_k=[1, 5],           # Eval top 1 and top 5.
    conv_num_filter_ratios=[2, 4],  # For CNN encoder, conv filter mutipler.
    conv_kernel_size=16,            # For CNN encoder, base kernel size.
    lstm_num_units=16,              # For LSTM/RNN, num units.
    num_predictions=10,          # Number of output predictions. Select top 10.
)

spec = recommendation.ModelSpec(
    input_spec=input_spec, model_hparams=model_hparams)
# Or:
spec = model_spec.get(
    'recommendation', input_spec=input_spec, model_hparams=model_hparams)
```
"""

from tensorflow_examples.lite.model_maker.core.data_util.recommendation_config import EncoderType
from tensorflow_examples.lite.model_maker.core.data_util.recommendation_config import Feature
from tensorflow_examples.lite.model_maker.core.data_util.recommendation_config import FeatureGroup
from tensorflow_examples.lite.model_maker.core.data_util.recommendation_config import FeatureType
from tensorflow_examples.lite.model_maker.core.data_util.recommendation_config import InputSpec
from tensorflow_examples.lite.model_maker.core.data_util.recommendation_config import ModelHParams
