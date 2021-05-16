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
loss, accuracy = model.evaluate(test_data)

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

DOCS['text_classifier'] = """APIs to train a text classification model.

Task guide:
https://www.tensorflow.org/lite/tutorials/model_maker_text_classification.
"""
# pylint: enable=line-too-long
