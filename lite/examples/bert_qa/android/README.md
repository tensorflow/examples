# TensorFlow Lite BERT QA Android Example Application


<img src="https://user-images.githubusercontent.com/67560900/122643946-37d0d380-d130-11eb-8e7c-f467b90cb0dd.mp4" width="400px" alt="Video">

## Overview

This is an end-to-end example of [BERT] Question & Answer application built with
TensorFlow 2.0, and tested on [SQuAD] dataset version 1.1. The demo app provides
48 passages from the dataset for users to choose from, and gives 5 most possible
answers corresponding to the input passage and query.

These instructions walk you through running the demo on an Android device. For an explanation of the source, see
[TensorFlow Lite BERT QA Android example](EXPLORE_THE_CODE.md).

### Model used

[BERT], or Bidirectional Encoder Representations from Transformers, is a method
of pre-training language representations which obtains state-of-the-art results
on a wide array of Natural Language Processing tasks.

This app uses [MobileBERT], a compressed version of [BERT] that runs 4x faster and
has 4x smaller model size.

For more information, refer to the [BERT github page][BERT].


## Build the demo using Android Studio

### Prerequisites

*   If you don't have already, install
    [Android Studio](https://developer.android.com/studio/index.html), following
    the instructions on the website.

*   Android Studio 3.2 or later.
    - Gradle 4.6 or higher.
    - SDK Build Tools 29.0.2 or higher.

*   You need an Android device or Android emulator and Android development
    environment with minimum API 21.

### Building

*   Open Android Studio, and from the Welcome screen, select `Open an existing
    Android Studio project`.

*   From the Open File or Project window that appears, navigate to and select
    the `bert_qa/android` directory from wherever you cloned the TensorFlow Lite
    sample GitHub repo.

*   You may also need to install various platforms and tools according to error
    messages.

*   If it asks you to use Instant Run, click Proceed Without Instant Run.

### Running

*   You need to have an Android device plugged in with developer options enabled
    at this point. See [here](https://developer.android.com/studio/run/device "Download Link")
    for more details on setting up developer devices.

*   If you already have Android emulator installed in Android Studio, select a
    virtual device with minimum API 15.

*   Click `Run` to run the demo app on your Android device.

#### Switch between inference solutions (Task library vs TFLite Interpreter)

This BERT QA Android reference app demonstrates two implementation
solutions:

(1)
[`lib_task_api`](https://github.com/SunitRoy2703/examples/tree/bertQa-android-task-lib/lite/examples/bert_qa/android/lib_task_api)
that leverages the out-of-box API from the
[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer);

(2)
[`lib_interpreter`](https://github.com/SunitRoy2703/examples/tree/bertQa-android-task-lib/lite/examples/bert_qa/android/lib_interpreter)
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

## Build the demo using gradle (command line)

### Building and Installing

*   Use the following command to build a demo apk:

```
cd lite/examples/bert_qa/android   # Folder for Android app.

./gradlew build
```

*   Use the following command to install the apk onto your connected device:

```
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Assets folder

_Do not delete the assets folder content_. If you explicitly deleted the files,
choose `Build -> Rebuild` to re-download the deleted model files into the assets
folder.

[BERT]: https://github.com/google-research/bert "Bert"
[SQuAD]: https://rajpurkar.github.io/SQuAD-explorer/ "SQuAD"
[MobileBERT]:https://tfhub.dev/tensorflow/tfjs-model/mobilebert/1 "MobileBERT"
