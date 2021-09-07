# OCR (Optical Character Recognition) Android sample.

OCR is the process of recognizing characters from images using computer vision
and machine learning techniques. This reference app demos how to use TensorFlow
Lite to do OCR. It uses a text detection model and a text recognition model as a
pipeline to recognize texts.

## Requirements

*   Android Studio 3.2 (installed on a Linux, Mac or Windows machine)
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
folder to `examples/lite/examples/optical_character_recognition/android`

### Step 3. Run the Android app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

To test the app, open the app called `TFL OCR` on your device. Re-installing the
app may require you to uninstall the previous installations.

For gradle CLI, running `./gradlew build` can create APKs for both solutions
under `app/build/outputs/apk`.

## Limitations

*   The current
    [text recognition model](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2)
    is trained using synthetic data with English letters and numbers, so only
    English is supported.

*   The models are not general enough for OCR in the wild (say, random images
    taken by a smartphone camera in a low lighting condition).

So we have chosen 3 Google product logos only to demonstrate how to do OCR with
TensorFlow Lite. If you are looking for a ready-to-use production-grade OCR
product, you should consider
[Google ML Kit](https://developers.google.com/ml-kit/vision/text-recognition).
ML Kit, which uses TFLIte underneath, should be sufficient for most OCR use
cases, but there are some cases where you may want to build your own OCR
solution with TFLite. Some examples are:

*   You have your own text detection/recognition TFLite models that you would
    like to use
*   You have special business requirements (i.e., recognizing texts that are
    upside down) and need to customize the OCR pipeline
*   You want to support languages not covered by ML Kit
*   Your target user devices don’t necessarily have Google Play services
    installed

## References

*   OpenCV text detection/recognition:
    https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
*   OCR TFLite community project by commnuity contributors @Tulasi123789 and
    @risingsayak https://github.com/tulasiram58827/ocr_tflite
*   OpenCV text detection:
    https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
*   Deep Learning based Text Detection Using OpenCV:
    https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
