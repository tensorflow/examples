# TensorFlow Lite Pose Estimation iOS Demo

### Overview

This is an app that continuously detects the human body parts in the frames seen
by your device's camera. These instructions walk you through building and
running the demo on an iOS device. Camera captures are discarded immediately
after use, nothing is stored or saved.

The app demonstrates how to use 3 models:
* Posenet
* Movenet Lightning
* Movenet Thunder

See this
[blog post](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html)
for a comparison between these models.

![Demo image](https://storage.googleapis.com/download.tensorflow.org/models/tflite/screenshots/posenet_ios_demo.gif)

## Build the demo using Xcode

### Prerequisites

*   [Xcode](https://developer.apple.com/xcode/) 12.5 or later
*   A valid Apple Developer ID
*   Real iOS device with camera
*   iOS version 12.4 or above
*   Xcode command line tools (to install, run `xcode-select --install`)
*   CocoaPods (to install, run `sudo gem install cocoapods`)

### Build and run the app

1.  Clone the TensorFlow examples GitHub repository to your computer to get the
    demo application.

    ```
    git clone https://github.com/tensorflow/examples
    ```

1.  Install the pod to generate the workspace file:

    ```
    cd examples/lite/examples/pose_estimation/ios && pod install
    ```

    Note: If you have installed this pod before and that command doesn't work,
    try `pod update`. At the end of this step you should have a directory called
    `PoseEstimation.xcworkspace`.

1.  Open the project in Xcode with the following command:

    ```
    open PoseEstimation.xcworkspace
    ```

    This launches Xcode and opens the `PoseEstimation` project.

1.  In Menu bar, select `Product` &rarr; `Destination` and choose your physical
    device.

1.  In Menu bar, select `Product` &rarr; `Run` to install the app on your
    device.

### Model used

The pose estimation is downloaded by `RunScripts/download_models.sh`. The script
is run automatically during the Xcode built process.

If you explicitly want to download the models, you can download them from here:

*   [Posenet](https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite)
*   [Movenet Lightning](https://tfhub.dev/google/movenet/singlepose/lightning/)
*   [Movenet Thunder](https://tfhub.dev/google/movenet/singlepose/thunder/)
