# TensorFlow Lite Image Classification iOS Example Application

**iOS Versions Supported:** iOS 12.0 and above.
**Xcode Version Required:** 10.0 and above

## Overview

This is a camera app that continuously classifies whatever it sees from your device's back camera, using a quantized MobileNet model. These instructions walk you through building and running the demo on an iOS device.

The model files are downloaded via scripts in Xcode as part of the build process. You don't need to perform any additional steps to download TFLite models into the project.

<!-- TODO(b/124116863): Add app screenshot. -->

## Prerequisites

* You must have Xcode installed

* You must have a valid Apple Developer ID

* The demo app requires a camera and must be executed on a real iOS device. You can build it and run with the iPhone Simulator but the app raises a camera not found exception.

* You don't need to build the entire TensorFlow library to run the demo, it uses CocoaPods to download the TensorFlow Lite library.

* You'll also need the Xcode command-line tools:

```xcode-select --install```

If this is a new install, you will need to run the Xcode application once to agree to the license before continuing.

## Building the iOS Demo App
1. Install CocoaPods if you don't have it.
```sudo gem install cocoapods```


2. Install the pod to generate the workspace file:
```cd examples/image_classification/ios```
```pod install```
If you have installed this pod before and that command doesn't work, try
```pod update```
At the end of this step you should have a file called ```ImageClassification.xcworkspace```

3. Open the project in Xcode by typing this on the command line:
```open ImageClassification.xcworkspace```
This launches Xcode if it isn't open already and opens the ```ImageClassification``` project.

4. Please change the bundle identifier to a unique identifier and select your development team in **'General->Signing'** before building the application if you are using an iOS device.

5. Build and run the app in Xcode.
You'll have to grant permissions for the app to use the device's camera. Point the camera at various objects and enjoy seeing how the model classifies things!

### Additional Note
_Please do not delete the empty references_ to the .tflite and .txt files after you clone the repo and open the project. These references will be fulfilled once the model and label files are downloaded when the application is built and run for the first time. If you delete the references to them, you can still find that the .tflite and .txt files are downloaded to the Model folder, the next time you build the application. You will have to add the references to these files in the bundle separately in that case.

## Model Used

This app uses [MobileNet](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) v1 model trained on [ImageNet](http://www.image-net.org/) 1000 classes. It takes input images of dimension 224 X 224 X 3. The model will be downloaded to the app as part of the build process. If required you can download the TFLite models for image classification [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md).

## iOS App Details

This app is written in both Swift and Objective C. All application functionality, image processing and results formatting is developed in Swift.
Objective C is used via bridging to make the TensorFlow Lite C++ framework calls.

## See Also

* [Apple Developer Guide on Importing Objective C into Swift](https://developer.apple.com/documentation/swift/imported_c_and_objective-c_apis/importing_objective-c_into_swift)
This documentation provides information on how to use code written in Objective C in an application written in Swift.
