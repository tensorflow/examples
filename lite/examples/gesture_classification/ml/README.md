# TensorFlow.js to TensorFlow Lite Model Conversion

This guide shows how you can go about converting the model trained with
TensorFlowJS to TensorFlow Lite FlatBuffers.

## Generate the TensorFlow.js model

To know more about generating the gesture classification tfjs model, go
[here](../web/README.md).

After following the steps below you will have `model.tflite` file downloaded,
after which you can proceed with running [Android Example](../android/README.md)
or [iOS Example](../ios/README.md).

## Colab notebook to convert the TensorFlow.js model to TensorFlow Lite

You can convert the model using this
[Colab notebook](https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/gesture_classification/ml/tensorflowjs_to_tflite_colab_notebook.ipynb)
and don't need to install any dependencies in your system.

Follow the steps in the notebook and you'll be asked to upload the `model.json`
and `model.weights.bin` files to Colab. They are the TensorFlow.js model that
you have generated with the [web app](../web/README.md).

Run all steps in-order. At the end, `model.tflite` file will be downloaded.

## Requirements

* TensorFlow.js 3.8.0 or above
* TensorFlow 2.5.0 or above

## Conversion Advanced Information

### Deserializing the Keras model from TensorFlow.js

To begin with, we have to convert the sequential block trained with
TensorFlow.js to Keras H5. This is done by passing the JSON model configuration
and the binary weights files downloaded from TensorFlow.js to the following
converter which generates a tf.Keras equivalent H5 model. This is achieved by
employing tfjs-converter's load_keras_model method which is going to deserialize
the Keras objects.

### Merging the base model and the classification block

This step involves loading the aforementioned custom classification block that
was just generated and then passing the output of the base model's intermediate
Depthwise Convolutional Layer's activation as input to the top classification
block.

```python

...
layer = self.base_model.get_layer('<insert the intermediate layer name here>') # e.g., conv_pw_13_relu
model = Model(inputs=self.base_model.input, outputs=self.top_model(layer.output))
...

```

### Generating the TensorFlow Lite model

After obtaining the Keras model, the TensorFlow Lite Converter is used to
convert the model from Keras to TFLite FlatBuffers.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

All of the above actions can be performed by running `convert.py`

## References

To know more about converting TensorFlow models from a SavedModel, Keras H5,
Session Bundle, Frozen Model or TensorFlow Hub module to a web-friendly format
and exporting a tfjs model to SavedModel, visit
[here](https://github.com/tensorflow/tfjs-converter).

## See Also

[Gesture Web Application](../web/README.md)
[Gesture Android Example](../android/README.md)
[Gesture iOS Example](../ios/README.md)
