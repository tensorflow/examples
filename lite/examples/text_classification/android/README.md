# TensorFlow Lite text classification sample

<img src="https://www.tensorflow.org/lite/examples/text_classification/images/screenshot.gif" width="400px" alt="Video">

## Overview

This is an end-to-end example of movie review sentiment classification built
with TensorFlow 2.0 (Keras API), and trained on [IMDB dataset](http://ai.stanford.edu/%7Eamaas/data/sentiment/) version 1.0. The demo app
processes input movie review texts, and classifies its sentiment into negative
(0) or positive (1).

These instructions walk you through the steps to train and test a simple text
classification model, export them to TensorFlow Lite format and deploy on a
mobile app.

## Model

See
[Text Classification with Movie Reviews](https://www.tensorflow.org/tutorials/keras/text_classification)
for a step-by-step instruction of building a simple text classification model.

## Build the demo using Android Studio
                                                                                                                                          
### Prerequisites

*   If you don't have already, install
    [Android Studio](https://developer.android.com/studio/index.html), following
    the instructions on the website.

*   Android Studio 3.2 or later.
    - Gradle 4.6 or higher.
    - SDK Build Tools 29.0.2 or higher.

*   You need an Android device or Android emulator and Android development
    environment with minimum API 21.

### Building

*   Open Android Studio, and from the Welcome screen, select `Open an existing
    Android Studio project`.

*   From the Open File or Project window that appears, navigate to and select
    the `text_classification/android` directory from wherever you cloned the
    TensorFlow Lite sample GitHub repo.

*   You may also need to install various platforms and tools according to error
    messages.

*   If it asks you to use Instant Run, click Proceed Without Instant Run.

### Running

*   You need to have an Android device plugged in with developer options enabled
    at this point. See [here](https://developer.android.com/studio/run/device)
    for more details on setting up developer devices.

*   If you already have an Android emulator installed in Android Studio, select
    a virtual device with API level higher than 15.

*   Click `Run` to run the demo app on your Android device.

## Build the demo using gradle (command line)

### Building and Installing

*   Use the following command to build a demo apk:

```
cd lite/examples/bert_qa/android   # Folder for Android app.

./gradlew build
```

*   Use the following command to install the apk onto your connected device:

```
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Assets folder

_Do not delete the assets folder content_. If you explicitly deleted the files,
choose `Build -> Rebuild` to re-download the deleted model files into the assets
folder.
