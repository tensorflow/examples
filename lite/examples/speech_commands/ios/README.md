# TensorFlow Lite Speech Command Recognition iOS Example Application

**iOS Versions Supported:** iOS 12.0 and above.
**Xcode Version Required:** 10.0 and above

## Overview

This app recognizes the specified set of voice commands from the microphone on the device. When the user speaks, commands for which the model is trained are identified.

These instructions will walk you through building and running the demo on an iOS device.

The model files are downloaded via scripts in Xcode when you build and run. You don't need to do any steps to download TFLite models into the project explicitly.

<!-- TODO(b/124116863): Add app screenshot. -->

## Prerequisites

* You must have Xcode installed

* You must have a valid Apple Developer ID

* The demo app requires the microphone and must be executed on a real iOS device. You can build it and run with the iPhone Simulator but the app raises a camera not found exception.

* You don't need to build the entire TensorFlow library to run the demo, it uses CocoaPods to download the TensorFlow Lite library.

* You'll also need the Xcode command-line tools:
```xcode-select --install```
If this is a new install, you will need to run the Xcode application once to agree to the license before continuing.
## Building the iOS Demo App

1. Install CocoaPods if you don't have it.
```sudo gem install cocoapods```

2. Install the pod to generate the workspace file:
```cd examples/speech_commands/ios/```
```pod install```
If you have installed this pod before and that command doesn't work, try
```pod update```
At the end of this step you should have a file called ```SpeechCommands.xcworkspace```

3. Open **SpeechCommands.xcworkspace** in Xcode.

4. Please change the bundle identifier to a unique identifier and select your development team in **'General->Signing'** before building the application if you are using an iOS device.

5. Build and run the app in Xcode.
You'll have to grant permissions for the app to use the device's microphone. The commands spoken by the user will be identified using the microphone input!

### Additional Note
_Please do not delete the empty references_ to the .tflite and .txt files after you clone the repo and open the project. These references will be fixed as the model and label files are downloaded when the application is built and run for the first time. If you delete the references, you can still find that the .tflite and .txt files are downloaded to the Model folder when you build the application. You will have to add the references manually to run the application after deleting.

## Model Used

You can find a detailed tutorial about training and running audio recognition model [here](https://www.tensorflow.org/tutorials/sequences/audio_recognition). The TFLite model can be downloaded [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/conv_actions_tflite.zip). The architecture of this model is based on the paper [Convolutional Neural Networks for Small-footprint Keyword Spotting](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf).

The percentage displayed is average command recognition over a window duration (1000ms).

## iOS App Details

The app is written entirely in Swift and uses the TensorFlow Lite
[Swift library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift)
for performing speech commands.

Note: Objective-C developers should use the TensorFlow Lite
[Objective-C library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/objc).
