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

Two alternative methods to install Model Maker library with its dependencies.

*   Install directly.

```shell
pip install git+https://github.com/tensorflow/examples.git#egg=tensorflow-examples[model_maker]
```

*   Clone the repo from the HEAD, and then install with pip.

```shell
git clone https://github.com/tensorflow/examples
cd examples
pip install .[model_maker]
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

Currently, we support image classification and text classification tasks and
provide demo code and colab for each of them in demo folder.

*   [Colab for image classification](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_image_classification.ipynb)
*   [Colab for text classification](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/model_maker_text_classification.ipynb)
