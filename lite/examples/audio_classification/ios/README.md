# TensorFlow Lite sound classification iOS example application

## Overview

This is an example application for [TensorFlow Lite](https://tensorflow.org/lite)
on iOS.

### Model

The model will be downloaded as part of the build process. Also, you can use
your own model generated on
[Teachable Machine](https://teachablemachine.withgoogle.com/train/audio). For an
explanation of training the model, see [Build sound classification models for
mobile apps with Teachable Machine and
TFLite](https://blog.tensorflow.org/2020/12/build-sound-classification-models-for-mobile-apps-with-teachable-machine-and-tflite.html).

### iOS app details

The app is written entirely in Swift and uses the TensorFlow Lite
[Swift library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/swift)
for performing sound classification.

## Requirements

*   Device with iOS 12.0 or above

*   Xcode 10.0 or above

*   Valid Apple Developer ID

*   Xcode command-line tools (run `xcode-select --install`)

*   [CocoaPods](https://cocoapods.org/) (run `sudo gem install cocoapods`)

If this is a new install, you will need to run the Xcode application once to
agree to the license before continuing.

Note:
The demo app requires `SelectTfOps` library which only works on a real iOS
device. You can build it and run with the iPhone Simulator, but the app will
raise a exception while initializing TensorFlow Lite runtime.

## Build and run

1.  Clone this GitHub repository to your workstation. `git clone
    https://github.com/tensorflow/examples.git`

2.  Install the pod to generate the workspace file: `cd
    examples/lite/examples/audio_classification/ios && pod install`

Note: If you have installed this pod before and that command doesn't work, try
`pod update`.

At the end of this step you should have a directory called
`SoundClassification.xcworkspace`.

1.  Open the project in Xcode with the following command: `open
    SoundClassification.xcworkspace`

This launches Xcode and opens the `SoundClassification` project.

1.  Select the `SoundClassification` project in the left hand navigation to open
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
_Do not delete the empty references_ to the .tflite and .txt files after you
clone the repo and open the project. These references will be fulfilled once the
model and label files are downloaded when the application is built and run for
the first time. If you delete the references to them, you can still find that
the .tflite and .txt files are downloaded to the Model folder, the next time you
build the application. You will have to add the references to these files in the
bundle separately in that case.
