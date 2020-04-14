# Style Transfer iOS sample

![Screenshot](https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/architecture.png)

## Requirements

*   Xcode 10.3 (installed on a Mac machine)
*   An iOS Simulator or device running iOS 10 or above
*   Xcode command-line tools (run `xcode-select --install`)
*   CocoaPods (run `sudo gem install cocoapods`)

## Build and run

1.  Clone the TensorFlow examples GitHub repository to your computer to get the
    demo application. `git clone https://github.com/tensorflow/examples` 1.
    Install the pod to generate the workspace file: `cd
    examples/lite/examples/style_transfer/ios && pod install` Note: If you have
    installed this pod before and that command doesn't work, try `pod update`.
    At the end of this step you should have a directory called
    `StyleTransfer.xcworkspace`.
1.  Open the project in Xcode with the following command:
    `open StyleTransfer.xcworkspace`<br/>
    This launches Xcode and opens the `StyleTransfer` project.
    1.  Select `Product -> Run` to install the app on an iOS Simulator or a
        physical device.
