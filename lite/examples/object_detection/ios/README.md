# TensorFlow Lite Object Detection iOS Example Application

**iOS Versions Supported:** iOS 12.0 and above.
**Xcode Version Required:** 10.0 and above

## Overview

This is a camera app that continuously detects the objects (bounding boxes and classes) in the frames seen by your device's back camera, using a quantized [MobileNet SSD](https://github.com/tensorflow/models/tree/master/research/object_detection) model trained on the [COCO dataset](http://cocodataset.org/). These instructions walk you through building and running the demo on an iOS device.

The model files are downloaded via scripts in Xcode when you build and run. You don't need to do any steps to download TFLite models into the project explicitly.

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
```cd therepowithnoname/examples/object_detection/ios/```
```pod install```
  If you have installed this pod before and that command doesn't work, try
```pod update```
At the end of this step you should have a file called ```ObjectDetection.xcworkspace```

3. Open **ObjectDetection.xcworkspace** in Xcode.

4. Please change the bundle identifier to a unique identifier and select your development team in **'General->Signing'** before building the application if you are using an iOS device.

5. Build and run the app in Xcode.
You'll have to grant permissions for the app to use the device's camera. Point the camera at various objects and enjoy seeing how the model classifies things!

### Note
_Please do not delete the empty references_ to the .tflite and .txt files after you clone the repo and open the project. These references will be fulfilled once the model and label files are downloaded when the application is built and run for the first time. If you delete the references to them, you can still find that the .tflite and .txt files are downloaded to the Model folder, the next time you build the application. You will have to add the references to these files in the bundle separately in that case.

## Model Used

This app uses a MobileNet SSD model trained on [COCO dataset](http://cocodataset.org/). The input image size required is 300 X 300 X 3. You can download the model [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip). You can find more information on the research on object detection [here](https://github.com/tensorflow/models/tree/master/research/object_detection).

## iOS App Details

This app is written in both Swift and Objective C. All application functionality, image processing and results formatting is developed in Swift.
Objective C is used via bridging to make the TensorFlow Lite C++ framework calls.
