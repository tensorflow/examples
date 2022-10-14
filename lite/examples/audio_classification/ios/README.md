# TensorFlow Lite audio classification iOS example application

## Overview

This is an example application for [TensorFlow Lite](https://tensorflow.org/lite)
on iOS.

### Model

The model will be downloaded as part of the build process. There are two models
included in this sample app: *
[YAMNet](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) is
a general purpose audio classification model that can detects 521 different type
of sounds. *
[Speech command](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition)
is a demonstrative model that can recognize a handful of single-word audio
command.

Also, you can use your own model generated on
[Teachable Machine](https://teachablemachine.withgoogle.com/train/audio) or
[Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/audio_classification).

### iOS app details

The app is written entirely in Swift and uses the TensorFlow Lite
[Swift library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift)
for performing sound classification.

## Requirements

*   Device with iOS 12.0 or above

*   Xcode 13.0 or above

*   Xcode command-line tools (run `xcode-select --install`)

*   [CocoaPods](https://cocoapods.org/) (run `sudo gem install cocoapods`)

*   (Optional) Valid Apple Developer ID. If you don't have one, you can run the
    sample app on an iOS Simulator.

If this is a new install, you will need to run the Xcode application once to
agree to the license before continuing.

## Build and run

1.  Clone this GitHub repository to your workstation. `git clone
    https://github.com/tensorflow/examples.git`

2.  Install the pod to generate the workspace file: `cd
    examples/lite/examples/sound_classification/ios && pod install`

Note: If you have installed this pod before and that command doesn't work, try
`pod update`.

At the end of this step you should have a directory called
`AudioClassification.xcworkspace`.

1.  Open the project in Xcode with the following command: `open
    AudioClassification.xcworkspace`

This launches Xcode and opens the `AudioClassification` project. You can run the
app on an iOS Sumi

Follow these steps to run the sample app on a physical device.

1.  Select the `AudioClassification` project in the left hand navigation to open
    the project configuration. In the **Signing** section of the **General**
    tab, select your development team from the dropdown.

2.  In order to build the project, you must modify the **Bundle Identifier** in
    the **Identity** section so that it is unique across all Xcode projects. To
    create a unique identifier, try adding your initials and a number to the end
    of the string.

3.  With an iOS device connected, build and run the app in Xcode.

You'll have to grant permissions for the app to use the device's camera. Point
the camera at various objects and enjoy seeing how the model classifies things!

## Model references

*Do not delete the empty references* to the .tflite files after you clone the
repo and open the project. These references will be fulfilled once the model and
label files are downloaded when the application is built and run for the first
time. If you delete the references to them, you can still find that the .tflite
and .txt files are downloaded to the Model folder, the next time you build the
application. You will have to add the references to these files in the bundle
separately in that case.
