# TensorFlow Lite Image Classification Demo

### Overview

This is a camera app that continuously classifies the objects in the frames
seen by your device's back camera, with the option to use a quantized
[MobileNet V1](https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224_quantized/1/metadata/1),
[EfficientNet Lite0](https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/int8/2),
[EfficientNet Lite1](https://tfhub.dev/tensorflow/lite-model/efficientnet/lite1/int8/2),
or
[EfficientNet Lite2](https://tfhub.dev/tensorflow/lite-model/efficientnet/lite2/int8/2)
model trained on Imagenet (ILSVRC-2012-CLS). These instructions
walk you through building and running the demo on an Android device.

The model files are downloaded via Gradle scripts when you build and run the
app. You don't need to do any steps to download TFLite models into the project
explicitly.

This application should be run on a physical Android device.

![App example showing UI controls. Result is espresso.](screenshot1.jpg?raw=true "Screenshot with controls")

![App example without UI controls. Result is espresso.](screenshot2.jpg?raw=true "Screenshot without controls")


## Build the demo using Android Studio

### Prerequisites

* The **[Android Studio](https://developer.android.com/studio/index.html)**
    IDE (Android Studio 2021.2.1 or newer). This sample has been tested on
    Android Studio Chipmunk

* A physical Android device with a minimum OS version of SDK 23 (Android 6.0 -
    Marshmallow) with developer mode enabled. The process of enabling developer
    mode may vary by device.

### Building

* Open Android Studio. From the Welcome screen, select Open an existing
    Android Studio project.

* From the Open File or Project window that appears, navigate to and select
    the tensorflow-lite/examples/image_classification/android directory.
    Click OK.

* If it asks you to do a Gradle Sync, click OK.

* With your Android device connected to your computer and developer mode
    enabled, click on the green Run arrow in Android Studio.

### Models used

Downloading, extraction, and placing the models into the assets folder is
managed automatically by the download.gradle file.
