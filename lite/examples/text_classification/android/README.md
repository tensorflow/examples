# TensorFlow Lite text classification sample

## Overview

This is an end-to-end example of movie review sentiment classification built
with TensorFlow 2.0 (Keras API), and trained on IMDB dataset. The demo app
processes input movie review texts, and classifies its sentiment into negative
(0) or positive (1).

These instructions walk you through the steps to train and test a simple text
classification model, export them to TensorFlow Lite format and deploy on a
mobile app.

## Model

See [Text Classification with Movie Reviews](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
for a step-by-step instruction of building a simple text classification model.

## Android app

Follow the steps below to build and run the sample Android app.

### Requirements

* Android Studio 3.2 or later.
Install instructions can be found on [Android Studio](https://developer.android.com/studio/index.html)
website.

* An Android device or an Android emulator and with API level higher than 15.

### Building

* Open Android Studio, and from the Welcome screen,
select `Open an existing Android Studio project`.

* From the Open File or Project window that appears,
navigate to and select the `text_classification/android` directory
from wherever you cloned the TensorFlow Lite sample GitHub repo.

* You may also need to install various platforms and tools according to
error messages.

* If it asks you to use Instant Run, click Proceed Without Instant Run.

### Running

* You need to have an Android device plugged in with developer options enabled
at this point. See [here](https://developer.android.com/studio/run/device)
for more details on setting up developer devices.

* If you already have an Android emulator installed in Android Studio,
select a virtual device with API level higher than 15.

* Click `Run` to run the demo app on your Android device.

