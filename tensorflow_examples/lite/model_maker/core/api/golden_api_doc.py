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
"""Golden APIs doc."""

DOCS = {}

# pylint: disable=line-too-long
DOCS[''] = """
Public APIs for TFLite Model Maker, a transfer learning library to train custom TFLite models.

You can install the package with

```bash
pip install tflite-model-maker
```

Typical usage of Model Maker is to create a model in a few lines of code, e.g.:

```python
# Load input data specific to an on-device ML app.
data = DataLoader.from_folder('flower_photos/')
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(train_data)

# Evaluate the model.
accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='/tmp/')
```

For more details, please refer to our guide:
https://www.tensorflow.org/lite/guide/model_maker.
""".lstrip()

DOCS['audio_classifier'] = """APIs to train an audio classification model.

Tutorial:
https://colab.research.google.com/github/googlecodelabs/odml-pathways/blob/main/audio_classification/colab/model_maker_audio_colab.ipynb

Demo code:
https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/audio_classification_demo.py
"""

DOCS['config'] = 'APIs for the config of TFLite Model Maker.'

DOCS['image_classifier'] = """APIs to train an image classification model.

Task guide:
https://www.tensorflow.org/lite/tutorials/model_maker_image_classification.
"""

DOCS['model_spec'] = 'APIs for the model spec of TFLite Model Maker.'

DOCS['object_detector'] = 'APIs to train an object detection model.'

DOCS['question_answer'] = """
APIs to train a model that can answer questions based on a predefined text.

Task guide:
https://www.tensorflow.org/lite/tutorials/model_maker_question_answer.
""".lstrip()

DOCS['recommendation'] = """APIs to train an on-device recommendation model.

Demo code:
https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/demo/recommendation_demo.py
"""

DOCS['recommendation.spec'] = """
APIs for recommendation specifications.

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
""".lstrip()

DOCS['text_classifier'] = """APIs to train a text classification model.

Task guide:
https://www.tensorflow.org/lite/tutorials/model_maker_text_classification.
"""
# pylint: enable=line-too-long
