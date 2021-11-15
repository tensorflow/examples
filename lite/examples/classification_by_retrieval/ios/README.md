# TensorFlow Lite classification by retrieval training iOS example application

## Overview

This is an example application for
[TensorFlow Lite](https://tensorflow.org/lite) on iOS. It uses
[Classification-by-Retrieval](../README.md) to train classification models out
of a small sample of user-provided images.

These instructions walk you through building and running the demo on an iOS
device.

| ![Demo of the iOS app](https://storage.googleapis.com/download.tensorflow.org/models/tflite_support/files/ios_app.gif) |
| ------ |

## Requirements

*   Device with iOS 14.0 or above

*   Xcode 12.5 or above

*   Valid Apple Developer ID

*   Xcode command-line tools (run `xcode-select --install`)

If this is a new install, you will need to run the Xcode application once to
agree to the license before continuing.

Note: The demo app requires a camera and must be executed on a real iOS device.
You can build it and run with the iPhone Simulator, but the app will raise a
`Camera not found` exception.

## Build and run

1.  Clone this GitHub repository to your workstation.

    ```
    $ git clone https://github.com/tensorflow/examples.git
    ```

1.  Change directory to the Classification-by-Retrieval example

    ```
    $ cd examples/lite/examples/classification_by_retrieval
    ```

1.  Build for a simulator with the following command:

    ```
    $ bazel build -c opt --config=ios_x86_64 ios:ImageClassifierBuilder
    ```

1.  To build for a device:
    1.  You'll first need to obtain a mobile provisioning profile from Apple
        ([documentation]).
    1.  Make a symlink to your profile like so:

        ```
        $ ln -s <path/to/your/profile.mobileprovision> ProvisioningProfile.mobileprovision
        ```

    1.  Uncomment all occurrences of `ProvisioningProfile.mobileprovision` in
        `ios/BUILD`.
    1.  Finally, build for a device with the following command:

        ```
        $ bazel build -c opt --config=ios_arm64 ios:ImageClassifierBuilder
        ```

You'll have to grant permissions for the app to use the device's camera and
Photo Library.

First you'll create a model out of albums of your Photo Library, then you'll be
able to see how it performs live: point the camera at things you trained your
model with and enjoy seeing how it classifies them!

[documentation]: https://developer.apple.com/documentation/appstoreconnectapi/profiles
