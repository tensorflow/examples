# Build your own aar

By default the app automatically downloads the needed aar files. But if you want
to build your own, just go ahead and run `./build_aar.sh`. This script will pull
in the necessary ops from [TensorFlow Text](https://www.tensorflow.org/text) and
build the aar for [Select TF operators](https://www.tensorflow.org/lite/guide/ops_select).

After compilation, a new file `tftext_tflite_flex.aar` is generated. Replace the
one in app/libs/ folder and re-build the app.

By default, the script builds only for `android_x86_64`. You can change it to 
`android_x86`, `android_arm` or `android_arm64`.

Note that you still need to include the standard `tensorflow-lite` aar in your 
gradle file.

