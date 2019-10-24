# How to Build Smart Reply Demo App.

## How to Build with Android Studio (GUI)

### Prerequisites

*   If you don't have already, install
    [Android Studio](https://developer.android.com/studio/index.html), following
    the instructions on the website.

*   Android Studio 3.2 or later.

*   You need an Android device or Android emulator and Android development
    environment with minimum API 15.

### Building and Run with Android Studio.

*   Open Android Studio, and from the Welcome screen, select `Open an existing
    Android Studio project`.

*   From the Open File or Project window that appears, navigate to and select
    the `lite/examples/smart_reply/android` directory from wherever you cloned
    the TensorFlow Lite sample GitHub repo.

*   You may also need to install various platforms and tools according to error
    messages.

*   Select menu `Build -> Make Project` to build the app. (Ctrl+F9, depending on
    your version).

*   Click menu `Run -> Run 'app'`. (Shift+F10, depending on your version)

## How to Build App with Gradle (Command Line).

### Step 1. Download pre-built AAR package containing custom ops.

The Smart Reply demo app contains custom ops to achieve machine learning
feature. It calls C++ ops through JNI that preprocesses the text from raw input.

We have pre-built [AAR package](https://storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply/smartreply_runtime_aar.aar)
  released, and it contains those custom ops that will be linked dynamically in
  the app. The corresponding smart reply [TF Lite model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply/smartreply.tflite)
  is also provided.

Change directory to Folder `android`, and download AAR:

```
cd lite/examples/smart_reply/android  # Your android folder

# Run gradle to download.
./gradlew :app:downloadAAR :app:downloadLiteModel
```

The above command downloads prebuilt AAR package with custom ops and TFLite
  model. You will have AAR package `smartreply_*_.aar` in folder `apps/libs`
  and `smartreply_*_.aar` in `apps/src/main/assets` and `apps/libs/cc/testdata`.

### Step 2. Use Gradle to build the app.

Then, use Gradle to build the Android app. Gradle will download dependencies and
finish building process. It creates android app under `app/build/outputs/apk`
directory.

```
./gradlew build
```

*Note that*: Either system environment variable `ANDROID_HOME` or `sdk.dir` in
  file `local.properties` for gradle should be set to Android SDK path.

### Step 3. Install apk.

Once packages of `debug` and `release` versions are created. You can install
this app on your phone or use emulator for debugging.

The following command uses `adb` to install the app to an Android phone or
  emulator connected.

```
adb install app/build/outputs/apk/debug/app-debug.apk
```

Then, you may be able to play with the app.

## Optional: How to build AAR package from source code.

If you changed C++ ops or JNI from source code, you may want to build AAR from
  source.

The procedure is to 1) build AAR package containing JNI (.so) lib, and 2) copy
  to Folder `libs`.

### Require: Bazel installed https://bazel.build/.

First, you need to set environment variables ANDROID_HOME and ANDROID_NDK_HOME
  for Android SDK and NDK respectively.

```# Notes: Depend on YOUR OWN installation. For example.
export ANDROID_HOME=$HOME/Android/Sdk
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/20.0.5594570
```

Use Bazel to build AAR package from JNI source code and include .so lib inside:

```
cd app

bazel build libs/cc:smartreply_runtime_aar
```

By default, it builds ops for multiple cpus (with options: `--fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a`).

From your bazel root folder, copy AAR package to `libs` folder.

```
cp bazel-bin/libs/cc/smartreply_runtime_aar.aar libs/smartreply_runtime_aar.aar
```

If you want to build optimized package you may add options `-c opt` or
  selectively choose some option in `--fat_apk_cpu`. For example,

```
bazel build -c opt --fat_apk_cpu=arm64-v8a,armeabi-v7a cc:smartreply_runtime_aar
```
