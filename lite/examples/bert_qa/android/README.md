# TensorFlow Lite BERT Question & Answer Demo Application

### Overview

This is an end-to-end example of BERT Question & Answer application built with
TensorFlow 2.0, and tested on SQuAD dataset. The demo app provides 48 passages
from the dataset for users to choose from, and gives 5 most possible answers
corresponding to the input passage and query. These instructions walk you
through building and running the demo on an Android device.

The model files are downloaded via Gradle scripts when you build and run
the app. You don't need to do any steps to download TFLite models into
the project explicitly.

This application should be run on a physical Android device.

![App example UI.](screenshot.png?raw=true "Screenshot QA screen")

## Build the demo using Android Studio

### Prerequisites

* The **[Android Studio](https://developer.android.com/studio/index.html)**
  IDE (Android Studio 2021.2.1 or newer). This sample has been tested on
  Android Studio Chipmunk.

* A physical Android device with a minimum OS version of SDK 23 (Android 6.0 -
  Marshmallow) with developer mode enabled. The process of enabling developer
  mode may vary by device.

### Building

* Open Android Studio. From the Welcome screen, select Open an existing
  Android Studio project.

* From the Open File or Project window that appears, navigate to and select the
  tensorflow-lite/examples/bert_qa/android directory. Click OK.

* If it asks you to do a Gradle Sync, click OK.

* With your Android device connected to your computer and developer mode
enabled, click on the green Run arrow in Android Studio.

### Models used

Downloading, extraction, and placing the models into the assets folder is
managed automatically by the download.gradle file.
