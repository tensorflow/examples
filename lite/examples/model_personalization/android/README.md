# TensorFlow Lite Model Personalization Demo

### Overview

This is a camera app that continuously classifies the objects in the frames seen
by your device's back camera. This example illustrates a way of personalizing a
TFLite model on-device without sending any data to the server. It can be adapted
for various tasks and models. These instructions walk you through building and
running the demo on an Android device.

The model files are downloaded via Gradle scripts when you build and run the
app. You don't need to do any steps to download TFLite models into the project
explicitly.

This application should be run on a physical Android device.

![App example showing UI controls. Training mode.](screenshot1.jpg?raw=true
"Training mode")

![App example without UI controls. Inference mode.](screenshot2.jpg?raw=true
"Inference mode")

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
  tensorflow-lite/examples/model_personalization/android directory. Click OK.

* If it asks you to do a Gradle Sync, click OK.

* With your Android device connected to your computer and developer mode
  enabled, click on the green Run arrow in Android Studio.

### Models used

Downloading, extraction, and placing the models into the assets folder is
managed automatically by the download.gradle file.

### Generate your model

To generate or customize your model you can read [here](../README.md).
