# TensorFlow Lite image classification Android example application

## Overview

This is an example application for [TensorFlow Lite](https://tensorflow.org/lite)
on Android. It uses
[Image classification](https://www.tensorflow.org/lite/models/image_classification/overview)
to continuously classify whatever it sees from the device's back camera.
Inference is performed using the TensorFlow Lite Java API. The demo app
classifies frames in real-time, displaying the top most probable
classifications. It allows the user to choose between a floating point or
[quantized](https://www.tensorflow.org/lite/performance/post_training_quantization)
model, select the thread count, and decide whether to run on CPU, GPU, or via
[NNAPI](https://developer.android.com/ndk/guides/neuralnetworks).

These instructions walk you through building and
running the demo on an Android device. For an explanation of the source, see
[TensorFlow Lite Android image classification example](https://www.tensorflow.org/lite/models/image_classification/android).

<!-- TODO(b/124116863): Add app screenshot. -->

### Model
For details of the model used, visit [Image classification](https://www.tensorflow.org/lite/models/image_classification/overview).

Downloading, extracting, and placing the model in the assets folder is managed
automatically by download.gradle.

## Requirements

*   Android Studio 3.2 (installed on a Linux, Mac or Windows machine)

*   Android device in
    [developer mode](https://developer.android.com/studio/debug/dev-options)
    with USB debugging enabled

*   USB cable (to connect Android device to your computer)

## Build and run

### Step 1. Clone the TensorFlow examples source code

Clone the TensorFlow examples GitHub repository to your computer to get the demo
application.

```
git clone https://github.com/tensorflow/examples
```

Open the TensorFlow source code in Android Studio. To do this, open Android
Studio and select `Open an existing project`, setting the folder to
`examples/lite/examples/image_classification/android`

<img src="images/classifydemo_img1.png?raw=true" />

### Step 2. Build the Android Studio project

Select `Build -> Make Project` and check that the project builds successfully.
You will need Android SDK configured in the settings. You'll need at least SDK
version 23. The `build.gradle` file will prompt you to download any missing
libraries.

The file `download.gradle` directs gradle to download the two models used in the
example, placing them into `assets`.

<img src="images/classifydemo_img4.png?raw=true" style="width: 40%" />

<img src="images/classifydemo_img2.png?raw=true" style="width: 60%" />

<aside class="note"><b>Note:</b><p>`build.gradle` is configured to use
TensorFlow Lite's nightly build.</p><p>If you see a build error related to
compatibility with Tensorflow Lite's Java API (for example, `method X is
undefined for type Interpreter`), there has likely been a backwards compatible
change to the API. You will need to run `git pull` in the examples repo to
obtain a version that is compatible with the nightly build.</p></aside>

### Step 3. Install and run the app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

<img src="images/classifydemo_img5.png?raw=true" style="width: 60%" />

<img src="images/classifydemo_img6.png?raw=true" style="width: 70%" />

<img src="images/classifydemo_img7.png?raw=true" style="width: 40%" />

<img src="images/classifydemo_img8.png?raw=true" style="width: 80%" />

To test the app, open the app called `TFL Classify` on your device. When you run
the app the first time, the app will request permission to access the camera.
Re-installing the app may require you to uninstall the previous installations.

## Assets folder
_Do not delete the assets folder content_. If you explicitly deleted the
files, choose `Build -> Rebuild` to re-download the deleted model files into the
assets folder.
