# TensorFlow Lite Gesture Classification iOS Example

**iOS Versions Supported:** iOS 12.0 and above.
**Xcode Version Required:** 10.0 and above

## Overview
This is a camera app that continuously classifies the gestures that the user shows, through the front camera. The model used in this app can be trained using a webcam. The instructions to be followed to train the model and convert it to a TFLite model can be found [here](../web/README.md). These instructions walk you through building and running the demo on an iOS device.

<!-- TODO(b/124116863): Add app screenshot. -->

## Prerequisites

* You must have Xcode installed

* You must have a valid Apple Developer ID

* The demo app requires a camera and must be executed on a real iOS device. You can build it and run with the iPhone Simulator but the app raises a camera not found exception.

* You don't need to build the entire TensorFlow library to run the demo, it uses CocoaPods to download the TensorFlow Lite library

* You'll also need the Xcode command-line tools:
```xcode-select --install```
If this is a new install, you will need to run the Xcode application once to agree to the license before continuing.

## Building the iOS Demo App

1. Install CocoaPods if you don't have it.
```sudo gem install cocoapods```

2. Install the pod to generate the workspace file:
```cd therepowithnoname/examples/gesture_classification/ios```
```pod install```
If you have installed this pod before and that command doesn't work, try
```pod update```
At the end of this step you should have a file called ```GestureClassification.xcworkspace```

3. Open **GestureClassification.xcworkspace** in Xcode.

4. This app uses a model that can be trained using images it collects from webcam. The instructions to train and convert the model can be found [here](../web/README.md).
You should add the downloaded TFLite model (filename **"model.tflite"**) and labels file (filename **"labels.txt"**) to the iOS app bundle. You can do this by dragging and dropping these files to the **'Model'** folder in **'Project Navigator**'.
When you drag and drop the files, you will get the following dialogue. Please click on **Finish** to add the files.

5. Please change the bundle identifier to a unique identifier and select your development team in **'General->Signing'** before building the application if you are using an iOS device.

6. Build and run the app in Xcode.
You'll have to grant permissions for the app to use the device's camera. You can show the gestures you trained in step 4 and the app identifies them in realtime!

## Model Used
This app uses [MobileNet](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) model that is trained on 0.25 alpha and at an image size of 224 X 244 X 3.

## iOS App Overview
This app is written in both Swift and Objective C. All application functionality, image processing and results formatting is developed in Swift.
Objective C is used via bridging to make the TensorFlow Lite C++ framework calls.


## See Also

* [Apple Developer Guide on Importing Objective C into Swift](https://developer.apple.com/documentation/swift/imported_c_and_objective-c_apis/importing_objective-c_into_swift)
This documentation provides information on how to use code written in Objective C for an application written in Swift.

## See Also

[Gesture Web Application](../web/README.md)
[Gesture ML Script](../ml/README.md)
[Gesture Android Example](../android/README.md)

