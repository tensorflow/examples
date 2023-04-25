## Prerequisites
If you haven’t already, install [Android Studio](https://developer.android.com/studio/index.html), following the instructions on the website.

*	Android Studio 2022.2.1 or above.
*	An Android device or Android emulator with more than 4G memory

## Building and Running with Android Studio
*	Open Android Studio, and from the Welcome screen, select Open an existing Android Studio project.
*	From the Open File or Project window that appears, navigate to and select the `lite/examples/generative_ai/android` directory from wherever you cloned the TensorFlow Lite sample GitHub repo.
*	You may also need to install various platforms and tools according to error messages.
*	Rename the converted `.tflite model` to `autocomplete.tflite` and copy it into `app/src/main/assets/` folder
*	Select menu `Build -> Make Project` to build the app. (Ctrl+F9, depending on your version).
*	Click menu `Run -> Run 'app'`. (Shift+F10, depending on your version)

Alternatively, you can also use the [gradle wrapper](https://docs.gradle.org/current/userguide/gradle_wrapper.html#gradle_wrapper) to build it in the command line. Please refer to the [Gradle documentation](https://docs.gradle.org/current/userguide/command_line_interface.html) for more information.

## (Optional) Building the .aar file
By default the app automatically downloads the needed .aar files. But if you want to build your own, switch to `app/libs/build_aar/` folder run `./build_aar.sh`. This script will pull in the necessary ops from [TensorFlow Text](https://www.tensorflow.org/text) and build the aar for [Select TF operators](https://www.tensorflow.org/lite/guide/ops_select).

After compilation, a new file `tftext_tflite_flex.aar` is generated. Replace the
`.aar` file in `app/libs/` folder and re-build the app.

Note that you still need to include the standard `tensorflow-lite` aar in your 
gradle file.