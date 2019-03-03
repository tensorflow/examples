# TensorFlow.js to TensorFlow Lite Model Conversion
This guide shows how you can go about converting the model trained with TensorFlowJS to TensorFlow Lite FlatBuffers.

## Generate the TensorFlow.js model
To know more about generating the gesture classification tfjs model, go [here](../web/README.md).

After following the steps below you will have `model.tflite` file downloaded, after which you can proceed with running [Android Example](../android/README.md) or [iOS Example](../ios/README.md).

## Colab Notebook (Recommended)
You can convert the model using the colab notebook below and don't need to install any dependencies in your system.

Upload the notebook into [Google Colaboratory](http://colab.research.google.com/)

After you have opened colaboratory page, colab notebook needs to be imported. You have to choose `Upload` or `File > Upload` and  pick the **[ipynb file](./tensorflowjs_to_tflite_colab_notebook.ipynb)**

Run all steps in-order. At the end, `model.tflite` file will be downloaded.

## Run Script

You can run the Python script provided also as an alternative to using the colab notebook and convert the `tfjs model`  files into `model.tflite` file. Following steps show how this can be done.

## Requirements

* TensorFlow.js 0.6.4
* Keras 2.2.2
* TensorFlow 1.11.0

## Converter
### Setup

Install the Python dependencies by running

```
pip install --force-reinstall -r requirements.txt
```

### Usage

```bash
python convert.py --config_json_path model.json
```

| Required Options | Description
|---|---|
|`--config_json_path`     | Path to the TensorFlow.js weights manifest file containing the model architecture (model.json) |

| Optional Parameters | Description
|---|---|
|`--model_tflite`| Converted TFLite model file. The name of the TensorFlow Lite model to be exported. |
|`--weights_path_prefix`| Optional path to weights files (model.weights.bin). If not specified (`None`), will assume the prefix is the same directory as the dirname of `model_json` with name `model.weights.bin`|

This will export the TensorFlow Lite model ready for inference on a mobile device such as Android or iOS.


## Conversion Script Advanced Information

### Deserializing the Keras model from TensorFlow.js
To begin with, we have to convert the sequential block trained with TensorFlow.js to Keras H5. This is done by passing the JSON model configuration and the binary weights files downloaded from TensorFlow.js to the following converter which generates a tf.Keras equivalent H5 model.
This is achieved by employing tfjs-converter's load_keras_model method which is going to deserialize the Keras objects.

### Merging the base model and the classification block
This step involves loading the aforementioned custom classification block that was just generated and then passing the output of the base model's intermediate Depthwise Convolutional Layer's activation as input to the top classification block.

```python

...
layer = self.base_model.get_layer('<insert the intermediate layer name here>') # e.g., conv_pw_13_relu
model = Model(inputs=self.base_model.input, outputs=self.top_model(layer.output))
...

```

### Generating the TensorFlow Lite model

After obtaining the Keras H5 model, the session is cleared of all the global variables and reloaded again from the saved Keras model. Finally, the TensorFlow Lite Optimizing Converter or TOCO is used to convert the model from Keras to TFLite FlatBuffers.

```python
converter = tf.contrib.lite.TocoConverter.from_keras_model_file(keras_model_file)
tflite_model = converter.convert()
```

All of the above actions can be performed by running `convert.py`

## References

To know more about converting TensorFlow models from a SavedModel, Keras H5, Session Bundle, Frozen Model or TensorFlow Hub module to a web-friendly format and exporting a tfjs model to Keras H5, visit [here](https://github.com/tensorflow/tfjs-converter).

## See Also

[Gesture Web Application](../web/README.md)
[Gesture Android Example](../android/README.md)
[Gesture iOS Example](../ios/README.md)

