# Tensorflow Lite PoseNet Android Demo
### Overview
This is an on-device camera app that detects the key points of a person in real-
time with the camera activity using the PoseNet model. This app uses the PoseNet
library that handles image processing, invoking the model, and processing output
from the model.

## Build the demo using Android Studio
### Building
* Type the following command in the terminal to build:
  bazel build :posenet_app
* Run the app on an android phone. If the application does not open
automatically, use adb to start the application on device.
Following command can be used from adb command line to start the application:
 am start org.tensorflow.lite.examples.posenet/.CameraActivity
