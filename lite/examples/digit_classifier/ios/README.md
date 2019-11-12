# Digit Classifier iOS sample

<img src="https://storage.googleapis.com/khanhlvg-public.appspot.com/digit-classifier/screenshot_ios.png" />

## Requirements

*  Xcode 10.3 (installed on a Mac machine)
*  An iOS Simuiator running iOS 12 or above
*  Xcode command-line tools (run ```xcode-select --install```)
*  CocoaPods (run ```sudo gem install cocoapods```)

## Build and run

1. Clone the TensorFlow examples GitHub repository to your computer to get the
demo
application.<br/>
```git clone https://github.com/tensorflow/examples```
1. Install the pod to generate the workspace file:<br/>
```cd examples/lite/examples/image_classification/ios && pod install```<br/>
Note: If you have installed this pod before and that command doesn't work, try ```pod update```.<br/>
At the end of this step you should have a directory called ```DigitClassifier.xcworkspace```.
1. Open the project in Xcode with the following command:<br/>
```open DigitClassifier.xcworkspace```<br/>
This launches Xcode and opens the ```DigitClassifier``` project.
1. Select `Product -> Run` to install the app on an iOS Simulator or a physical
device.
