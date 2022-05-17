# TensorFlow Lite text classification sample

## Overview

This is an end-to-end example of movie review sentiment classification built
with TensorFlow 2.0 (Keras API), and trained on IMDB dataset. The demo app
processes input movie review texts, and classifies its sentiment into negative
(0) or positive (1).

These instructions walk you through the steps to train and test a simple text
classification model, export them to TensorFlow Lite format and deploy on a
mobile app.

## Model

See
[Text Classification with Movie Reviews](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
for a step-by-step instruction of building a simple text classification model.

## Android app

Follow the steps below to build and run the sample Android app.

### Requirements

*   Android Studio 4.2 or above. Install instructions can be found on
    [Android Studio](https://developer.android.com/studio/index.html) website.

*   An Android device or an Android emulator and with minimum API 21.

### Building

*   Open Android Studio, and from the Welcome screen, select `Open an existing
    Android Studio project`.

*   From the Open File or Project window that appears, navigate to and select
    the `text_classification/android` directory from wherever you cloned the
    TensorFlow Lite sample GitHub repo.

*   You may also need to install various platforms and tools according to error
    messages.

*   If it asks you to use Instant Run, click Proceed Without Instant Run.

### Running

*   You need to have an Android device plugged in with developer options enabled
    at this point. See [here](https://developer.android.com/studio/run/device)
    for more details on setting up developer devices.

*   If you already have an Android emulator installed in Android Studio, select
    a virtual device with minimum API 21.

*   Click `Run` to run the demo app on your Android device.

#### Switch between inference solutions (Task library vs TFLite Interpreter)

This Text Classification Android reference app demonstrates two implementation
solutions:

(1)
[`lib_task_api`](https://github.com/tensorflow/examples/tree/master/lite/examples/nl_classification/android/lib_task_api)
that leverages the out-of-box API from the
[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/text_classifier);

(2)
[`lib_interpreter`](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_interpreter)
that creates the custom inference pipleline using the
[TensorFlow Lite Interpreter Java API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_java).

The [`build.gradle`](app/build.gradle) inside `app` folder shows how to change
`flavorDimensions "tfliteInference"` to switch between the two solutions.

Inside **Android Studio**, you can change the build variant to whichever one you
want to build and run—just go to `Build > Select Build Variant` and select one
from the drop-down menu. See
[configure product flavors in Android Studio](https://developer.android.com/studio/build/build-variants#product-flavors)
for more details.

For gradle CLI, running `./gradlew build` can create APKs for both solutions
under `app/build/outputs/apk`.

*Note: If you simply want the out-of-box API to run the app, we recommend
`lib_task_api`for inference. If you want to customize your own models and
control the detail of inputs and outputs, it might be easier to adapt your model
inputs and outputs by using `lib_interpreter`.*
