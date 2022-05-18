# TensorFlow Lite Gesture Classification Android Example

### Overview

This app performs gesture classification on live camera feed and displays the results in real-time on the screen.

Application can run either on device or emulator.

<!-- TODO(b/124116863): Add app screenshot. -->

## Build the demo using Android Studio

### Prerequisites

* If you don't have already, install **[Android Studio](https://developer.android.com/studio/index.html)**, following the instructions on their website.
* You need an Android device and Android development environment with minimum API 21.
* Android Studio 4.2 or above.

* This is the  **[guide](../web/README.md)**  to generate TensorFlow.js model.
* Name the .tflite file as **model.tflite** and the subset of labels should be named as **labels.txt**.

### Building

* Open Android Studio, and from the Welcome screen, select Open an existing Android Studio project.

* From the Open File or Project window that appears, navigate to and select the tensorflow-lite/examples/gesture_classification/android directory from wherever you cloned the TensorFlow Lite sample GitHub repo. Click OK.

* If it asks you to do a Gradle Sync, click OK.

* You may also need to install various platforms and tools, if you get errors like "Failed to find target with hash string 'android-21'" and similar.
Click the Run button (the green arrow) or select Run > Run 'android' from the top menu. You may need to rebuild the project using Build > Rebuild Project.

* If it asks you to use Instant Run, click Proceed Without Instant Run.

* Also, you need to have an Android device plugged in with developer options enabled at this point. See **[here](https://developer.android.com/studio/run/device)** for more details on setting up developer devices.
* Read the following **[doc](../ml/README.md)** to generate TFLite model file.
* Copy the labels.txt and put into assets folder.
* Now once you get the TFLite and label file, put that into assets folder.

### Additional Note

Ensure that labels.txt and model.tflite files are added into the project which are downloaded from [web app](../web/README.md). Also ensure gesture_labels.txt file is not deleted or modified from assets folder.

## See Also

* [Gesture Web Application](../web/README.md)
* [Gesture ML Script](../ml/README.md)
* [Gesture iOS Example](../ios/README.md)
