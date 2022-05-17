# Style Transfer Android sample.

Artistic style transfer is an optimization technique used to take two images: a
content image and a style reference image (such as an artwork by a famous
painter) and blend them together so the output image looks like the content
image, but “painted” in the style of the style reference image.

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
*   [Style Transfer model for mobile](https://www.tensorflow.org/lite/models/style_transfer/overview)
*   [Train the style transfer model and export to TensorFlow Lite](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization#train-a-model-on-a-large-dataset-with-data-augmentation-to-run-on-mobile)
*   [Neural Style Transfer with TensorFlow](https://www.tensorflow.org/tutorials/generative/style_transfer)
*   [CameraX](https://developer.android.com/training/camerax)
