# TensorFlow Lite Recommendation Android Example Application

## Overview

This is an end-to-end example of recommendation application built with
TensorFlow 2.0, and the model is trained based on the public
[MovieLens dataset](https://grouplens.org/datasets/movielens/). The dataset and
model is used for research purpose. This demo is supposed to be a reference
architecture to demonstrate how to build and run on-device recommendation model
without in TF Lite talking to a server.

These instructions walk you through running the demo on an Android device.

## Build the demo using Android Studio

### Prerequisites

*   If you don't have already, install
    [Android Studio](https://developer.android.com/studio/index.html), following
    the instructions on the website.

*   Android Studio 3.2 or later.

*   You need an Android device or Android emulator and Android development
    environment with minimum API 15.

### Building

*   Open Android Studio, and from the Welcome screen, select `Open an existing
    Android Studio project`.

*   From the Open File or Project window that appears, navigate to and select
    the `recommendation/android` directory from wherever you cloned the
    TensorFlow Lite sample GitHub repo.

*   You may also need to install various platforms and tools according to error
    messages.

*   If it asks you to use Instant Run, click Proceed Without Instant Run.

### Running

*   You need to have an Android device plugged in with developer options enabled
    at this point. See [here](https://developer.android.com/studio/run/device)
    for more details on setting up developer devices.

*   If you already have Android emulator installed in Android Studio, select a
    virtual device with minimum API 15.

*   Click `Run` to run the demo app on your Android device.

## Build the demo using gradle (command line)

### Building and Installing

*   Use the following command to build a demo apk:

```
cd lite/examples/recommendation/android   # Folder for Android app.

./gradlew build
```

*   Use the following command to install the apk onto your connected device:

```
adb install -t app/build/outputs/apk/debug/app-debug.apk
```
