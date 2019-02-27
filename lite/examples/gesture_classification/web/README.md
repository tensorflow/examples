# Gesture Classification Web App
**Browsers Supported:** Chrome, Safari.

Note : Model download does not work in Firefox.

## Overview
This is a web app used to predict gesture from a webcam using transfer learning. We will use a pretrained truncated MobileNet model and train another model using the internal MobileNet activation to predict upto 8 different classes from the webcam defined by the user.

`index.html` can be opened directly inside a browser. No need to have a web server to run the web app.

## About the Model
To learn to classify different classes from the webcam in a reasonable amount of time, we will retrain, or fine-tune, a pretrained MobileNet model, using the internal activation (the output from an internal layer of MobileNet) as input to our new model.

To do this, we'll be having two models on the page.

One model will be the pretrained MobileNet model that is truncated to output the internal activations. We'll call this the "truncated MobileNet model". This model does not get trained after being loaded into the browser.

The second model will take as input the output of the internal activation of the truncated MobileNet model and will predict probabilities for each of the selected output classes which can be up, down, left, right, left click, right click, scroll up and scroll down. This is the model we'll actually train in the browser.

By using the internal activation of MobileNet, we can reuse the features that MobileNet has already learned to predict the 1000 classes of ImageNet with a relatively small amount of retraining.

## Model Specifications

The base model being used here is MobileNet with a width of .25 and input image size of 224 X 224. The width and the input size can be varied.
* Width: 0.25, 0.50, 0.75 or 1.0 (default: 0.25).
* Image size: 128, 160, 192 or 224 (default: 224).

You can choose to pick an intermediate depth wise convolutional layer such as `conv_pw_13_relu` by calling `getLayer('conv_pw_13_relu')`. Try choosing another layer and see how it affects model quality! You can use model.layers to print the layers of the model.

## Prerequisites

* This app requires webcam.

## Importing TensorFlow.js Library
```
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@VERSION"> </script>
```
You can change the version by replacing `VERSION` with appropriate version. eg: 0.13.0.

## How to use the App

1. Open index.html in your browser.

2. Collect adequate samples for each of the required gestures by clicking on their icons.

3. Set the parameters for training the model such as Learning rate, Batch size, Number of Epochs and Number of Hidden Units in the top sequential model.

4. Click on Train button to train the model. Wait for sometime until the model is trained.

5. Once the model is trained you can either choose to test or download the model. Upon downloading the model, totally 3 files will be generated which are weights manifest file (model.json), the binary weights file(model-weights.bin) and labels.txt for the chosen gestures.

## Conversion to TensorFlow Lite
  To carry on with the conversion process to TFLite model, visit [here](../ml/README.md).

## Additional instruction for Safari
Enable Develop Menu > WebRTC > Allow Media Capture on Insecure Sites

# See Also


[Gesture ML Script](../ml/README.md)
[Gesture Android Example](../android/README.md)
[Gesture iOS Application](../ios/README.md)

