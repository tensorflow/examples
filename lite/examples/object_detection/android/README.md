# TensorFlow Lite Object Detection Android Demo
### Overview
This is a camera app that continuously detects the objects (bounding boxes and classes) in the frames seen by your device's back camera, using a quantized [MobileNet SSD](https://github.com/tensorflow/models/tree/master/research/object_detection) model trained on the [COCO dataset](http://cocodataset.org/). These instructions walk you through building and running the demo on an Android device.

The model files are downloaded via Gradle scripts when you build and run. You don't need to do any steps to download TFLite models into the project explicitly.

Application can run either on device or emulator.

<!-- TODO(b/124116863): Add app screenshot. -->

## Build the demo using Android Studio

### Prerequisites

* If you don't have already, install **[Android Studio](https://developer.android.com/studio/index.html)**, following the instructions on the website.

* You need an Android device and Android development environment with minimum API 21.
* Android Studio 3.2 or later.

### Building
* Open Android Studio, and from the Welcome screen, select Open an existing Android Studio project.

* From the Open File or Project window that appears, navigate to and select the tensorflow-lite/examples/object_detection/android directory from wherever you cloned the TensorFlow Lite sample GitHub repo. Click OK.

* If it asks you to do a Gradle Sync, click OK.

* You may also need to install various platforms and tools, if you get errors like "Failed to find target with hash string 'android-21'" and similar.
Click the Run button (the green arrow) or select Run > Run 'android' from the top menu. You may need to rebuild the project using Build > Rebuild Project.

* If it asks you to use Instant Run, click Proceed Without Instant Run.

* Also, you need to have an Android device plugged in with developer options enabled at this point. See **[here](https://developer.android.com/studio/run/device)** for more details on setting up developer devices.


### Model used
Downloading, extraction and placing it in assets folder has been managed automatically by download.gradle.

If you explicitly want to download the model, you can download from **[here](http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)**. Extract the zip to get the .tflite and label file.


### Custom model used
This example shows you how to perform TensorFlow Lite object detection using a custom model.
* Clone the TensorFlow models GitHub repository to your computer.
```
git clone https://github.com/tensorflow/models/
```
* Build and install this repository.
```
cd models/research
python3 setup.py build && python3 setup.py install
```
* Download the MobileNet SSD trained on **[Open Images v4](https://storage.googleapis.com/openimages/web/factsfigures_v4.html)** **[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)**. Extract the pretrained TensorFlow model files.
* Go to `models/research` directory and execute this code to get the frozen TensorFlow Lite graph.
```
python3 object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path object_detection/samples/configs/ssd_mobilenet_v2_oid_v4.config \
  --trained_checkpoint_prefix <directory with ssd_mobilenet_v2_oid_v4_2018_12_12>/model.ckpt \
  --output_directory exported_model
```
* Convert the frozen graph to the TFLite model.
```
tflite_convert \
  --input_shape=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
  --allow_custom_ops \
  --graph_def_file=exported_model/tflite_graph.pb \
  --output_file=<directory with the TensorFlow examples repository>/lite/examples/object_detection/android/app/src/main/assets/detect.tflite
```
`input_shape=1,300,300,3` because the pretrained model works only with that input shape.

`allow_custom_ops` is necessary to allow TFLite_Detection_PostProcess operation.

`input_arrays` and `output_arrays` can be drawn from the visualized graph of the example detection model.
```
bazel run //tensorflow/lite/tools:visualize \
  "<directory with the TensorFlow examples repository>/lite/examples/object_detection/android/app/src/main/assets/detect.tflite" \
  detect.html
```

* Get `labelmap.txt` from the second column of **[class-descriptions-boxable](https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv)**.
* In `DetectorActivity.java` set `TF_OD_API_IS_QUANTIZED` to `false` and in `TFLiteObjectDetectionAPIModel.java` set `labelOffset` to `0`.


### Additional Note
_Please do not delete the assets folder content_. If you explicitly deleted the files, then please choose *Build*->*Rebuild* from menu to re-download the deleted model files into assets folder.
