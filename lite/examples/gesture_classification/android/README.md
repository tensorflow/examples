# TensorFlow Lite Gesture Classification

### Overview

This is a camera app that continuously classifies the gesture in the frames seen
by your device's front camera. These instructions walk you through building and
running the demo on an Android device.

The model files are downloaded via Gradle scripts when you build and run the
app. You don't need to do any steps to download TFLite models into the project
explicitly.

We trained the model with the following hand gesture images.
![Training images](gesture_training_image.png "Training images")

This application should be run on a physical Android device.

![App example showing UI controls. Result is down.](
gesture_classification_screen_shot.jpeg?raw=true "Screenshot with controls")

## Build the demo using Android Studio

### Prerequisites

* The **[Android Studio](https://developer.android.com/studio/index.html)**
  IDE (Android Studio 2021.2.1 or newer). This sample has been tested on Android
  Studio Chipmunk.

* A physical Android device with a minimum OS version of SDK 23 (Android 6.0 -
  Marshmallow) with developer mode enabled. The process of enabling developer
  mode may vary by device.

### Building

* Open Android Studio. From the Welcome screen, select Open an existing Android
  Studio project.

* From the Open File or Project window that appears, navigate to and select the
  tensorflow-lite/examples/gesture_classification/android directory. Click OK.

* If it asks you to do a Gradle Sync, click OK.

* With your Android device connected to your computer and developer mode
  enabled, click on the green Run arrow in Android Studio.

### Customize your model

* Read the following **[doc](../ml/README.md)** to generate TFLite model file.

* Add the metadata to the tflite model using this [Colab notebook](
https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/gesture_classification/android/adding_metadata_to_tflite_model_colab_notebook.ipynb)

* Copy the downloaded model into app/src/main/assets folder. Make sure the
  model is named model_metadata.tflite for this sample.
