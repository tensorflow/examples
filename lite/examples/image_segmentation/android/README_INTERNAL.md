# Image Segmentation Android sample.

The used model, DeepLab [https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html]
is a state-of-art deep learning model for semantic image segmentation,
where the goal is to assign semantic labels (e.g. person, dog, cat) to every pixel in the input
image.

## Requirements

*  Android Studio 4.2 or above (installed on a Linux, Mac or Windows machine)
*  An Android device, or an Android Emulator

## Build and run

### Step 1. Clone the TensorFlow examples source code

Clone the TensorFlow examples GitHub repository to your computer to get the
demo application.

```
git clone https://github.com/tensorflow/examples
```

### Step 2. Import the sample app to Android Studio

Open the TensorFlow source code in Android Studio. To do this, open Android
Studio and select `Import Projects (Gradle, Eclipse ADT, etc.)`, setting the
folder to `examples/lite/examples/image_segmentation/android`


### Step 3. Run the Android app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

To test the app, open the app called `TFL Image Segmentation` on your device.
Re-installing the app may require you to uninstall the previous installations.


## Resources used:

* Camera2: https://developer.android.com/reference/android/hardware/camera2/package-summary
* Canera2 base sample: https://github.com/android/camera-samples/tree/master/Camera2Formats
* TensorFlow Lite: https://www.tensorflow.org/lite
* ImageSegmentation model: https://www.tensorflow.org/lite/models/segmentation/overview
