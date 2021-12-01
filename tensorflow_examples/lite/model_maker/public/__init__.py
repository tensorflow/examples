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
"""Public APIs for TFLite Model Maker, a transfer learning library to train custom TFLite models.

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
"""

from tflite_model_maker import audio_classifier
from tflite_model_maker import config
from tflite_model_maker import image_classifier
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker import question_answer
from tflite_model_maker import recommendation
from tflite_model_maker import text_classifier

# Deprecated imports are kept for backward compatiblity and to be removed in
# future versions. Please refer to public APIs for replacement:
# https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker
# pylint: disable=g-bad-import-order
from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import configs
# pylint: enable=g-bad-import-order

__version__ = '0.3.4'
