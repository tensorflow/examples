# TFLite model customization

## Overview

The TFLite model customization library simplifies the process of adapting and
converting a TensorFlow neural-network model to particular input data when
deploying this model for on-device ML applications.

## End-to-End Example

For instance, it could have an end-to-end image
classfication example that utilizes this library with just 4 lines of
code, each of which representing one step of the overall process:

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

4.  Export to Tensorflow Lite model.

```python
model.export('flower_classifier.tflite', 'flower_label.txt')
```

## Notebook

Currently, we support image classification and text classification tasks and
provide demo code and colab for each of them in demo folder.

* [Colab for image classification](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_customization/demo/image_classification.ipynb)
* [Colab for text classfication](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_customization/demo/text_classification.ipynb)

