# Sound Classifier Android sample.


## Requirements

*   Android Studio 4.1 (installed on a Linux, Mac or Windows machine)
*   An Android device, or an Android Emulator

## Build and run

### Step 1. Clone the TensorFlow examples source code

Clone the TensorFlow examples GitHub repository to your computer to get the demo
application.

```
git clone https://github.com/tensorflow/examples
```

### Step 2. Import the sample app to Android Studio

Open the TensorFlow source code in Android Studio. To do this, open Android
Studio and select `Import Projects (Gradle, Eclipse ADT, etc.)`, setting the
folder to `examples/lite/examples/sound_classification/android`

### Step 3. Run the Android app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

To test the app, open the app called `TFL Sound Classifier` on your device.
Re-installing the app may require you to uninstall the previous installations.

## Resources used:

*   [TensorFlow Lite](https://www.tensorflow.org/lite)
*   [YAMNet audio classification model](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1)
