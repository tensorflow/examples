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

## Installation

There are two ways to install Model Maker.

*   Install a prebuilt pip package.

```shell
pip install tflite-model-maker
```

If you want to install nightly version, please follow the command:

```shell
pip install tflite-model-maker-nightly
```

*   Clone the source code from GitHub and install.

```shell
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
```

## End-to-End Example

For instance, it could have an end-to-end image classification example that
utilizes this library with just 4 lines of code, each of which representing one
step of the overall process. For more detail, you could refer to
[Colab for image classification](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_image_classification.ipynb).

1.   Load input data specific to an on-device ML app.

```python
data = ImageClassifierDataLoader.from_folder('flower_photos/')
```

2. Customize the TensorFlow model.

```python
model = image_classifier.create(data)
```

3. Evaluate the model.

```python
loss, accuracy = model.evaluate()
```

4.  Export to Tensorflow Lite model and label file in `export_dir`.

```python
model.export(export_dir='/tmp/')
```

## Notebook

Currently, we support image classification, text classification and question
answer tasks. Meanwhile, we provide demo code for each of them in demo folder.

*   [Overview for TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker)
*   [Colab for image classification](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_image_classification.ipynb)
*   [Colab for text classification](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_text_classification.ipynb)
*   [Colab for question answer](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_question_answer.ipynb)
