# TFLite Model Maker

## Overview

The TFLite Model Maker library simplifies the process of adapting and converting
a TensorFlow neural-network model to particular input data when deploying this
model for on-device ML applications.

## Requirements

*   Refer to
    [requirements.txt](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/requirements.txt)
    for dependent libraries that're needed to use the library and run the demo
    code.
*   Note that you might also need to install `sndfile` for Audio tasks.
On Debian/Ubuntu, you can do so by `sudo apt-get install libsndfile1`

## Installation

There are two ways to install Model Maker.

*   Install a prebuilt pip package:
    [`tflite-model-maker`](https://pypi.org/project/tflite-model-maker/).

```shell
pip install tflite-model-maker
```

If you want to install nightly version
[`tflite-model-maker-nightly`](https://pypi.org/project/tflite-model-maker-nightly/),
please follow the command:

```shell
pip install tflite-model-maker-nightly
```

*   Clone the source code from GitHub and install.

```shell
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
```

TensorFlow Lite Model Maker depends on TensorFlow
[pip package](https://www.tensorflow.org/install/pip). For GPU support, please
refer to TensorFlow's [GPU guide](https://www.tensorflow.org/install/gpu) or
[installation guide](https://www.tensorflow.org/install).

## End-to-End Example

For instance, it could have an end-to-end image classification example that
utilizes this library with just 4 lines of code, each of which representing one
step of the overall process. For more detail, you could refer to
[Colab for image classification](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_image_classification.ipynb).

*   Step 1. Import the required modules.

```python
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader
```

*   Step 2. Load input data specific to an on-device ML app.

```python
data = DataLoader.from_folder('flower_photos/')
```

*   Step 3. Customize the TensorFlow model.

```python
model = image_classifier.create(data)
```

*   Step 4. Evaluate the model.

```python
loss, accuracy = model.evaluate()
```

*   Step 5. Export to Tensorflow Lite model and label file in `export_dir`.

```python
model.export(export_dir='/tmp/')
```

## Notebook

Currently, we support image classification, text classification and question
answer tasks. Meanwhile, we provide demo code for each of them in demo folder.

*   [Overview for TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker)
*   [Python API Reference](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker)
*   [Colab for image classification](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_image_classification.ipynb)
*   [Colab for text classification](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_text_classification.ipynb)
*   [Colab for BERT question answer](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_question_answer.ipynb)
*   [Colab for object detection](https://www.tensorflow.org/lite/tutorials/model_maker_object_detection)
