# Text Searcher Android sample

This sample app demonstrates how to use the TensorFlow Lite Task Library's Text
Searcher API on Android. It works by using a model to embed the search query
into a high-dimensional vector representing the semantic meaning of the query.
Then it uses
[ScaNN (Scalable Nearest Neighbors)](https://github.com/google-research/google-research/tree/master/scann)
to search for similar items from a predefined database.

## Requirements

*   Android Studio 4.2 or above (installed on a Linux, Mac or Windows machine)
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
folder to `examples/lite/examples/style_transfer/android`

### Step 3. Run the Android app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

To test the app, open the app called `TFL Style Transfer` on your device.
Re-installing the app may require you to uninstall the previous installations.

## Resources used:

*   [TensorFlow Lite](https://www.tensorflow.org/lite)
*   [TensorFlow Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview)
*   [CNN/DailyMail non-anonymized summarization dataset](https://github.com/abisee/cnn-dailymail)
*   [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker)
