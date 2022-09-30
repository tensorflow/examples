# TensorFlow Lite Style Transfer Demo

### Overview

Artistic style transfer is an optimization technique used to take two images: a
content image and a style reference image (such as an artwork by a famous
painter) and blend them together so the output image looks like the content
image, but “painted” in the style of the style reference image. These
instructions walk you through building and running the demo on an Android
device.

The model files are downloaded via Gradle scripts when you build and run the
app. You don't need to do any steps to download TFLite models into the project
explicitly.

This application should be run on a physical Android device.

![Camera screen.](screenshot1.jpg?raw=true "Camera screen")

![Transformation screen.](screenshot2.jpg?raw=true "Transformation screen")

Test image
from [Pixabay](https://pixabay.com/photos/tiger-head-face-feline-wild-cat-2923186/)

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
  tensorflow-lite/examples/style_transfer/android directory. Click OK.

* If it asks you to do a Gradle Sync, click OK.

* With your Android device connected to your computer and developer mode
  enabled, click on the green Run arrow in Android Studio.

### Models used

Downloading, extraction, and placing the models into the assets folder is
managed automatically by the download.gradle file.

### Resources used

* [TensorFlow Lite](https://www.tensorflow.org/lite)
* [Train the style transfer model and export to TensorFlow Lite](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization#train-a-model-on-a-large-dataset-with-data-augmentation-to-run-on-mobile)
* [Neural Style Transfer with TensorFlow](https://www.tensorflow.org/tutorials/generative/style_transfer)
* [CameraX](https://developer.android.com/training/camerax)
