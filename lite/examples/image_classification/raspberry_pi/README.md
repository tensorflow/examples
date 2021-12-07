# TensorFlow Lite Python image classification example with Raspberry Pi.

This example uses [TensorFlow Lite](https://tensorflow.org/lite) with Python
on a Raspberry Pi to perform real-time image classification using images
streamed from the camera.

At the end of this page, there are extra steps to accelerate the example using
the Coral USB Accelerator, which increases the inference speed by ~10x.


## Set up your hardware

Before you begin, you need to [set up your Raspberry Pi](
https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up) with
Raspberry Pi OS (preferably updated to Buster).

You also need to [connect and configure the Pi Camera](
https://www.raspberrypi.org/documentation/configuration/camera.md) if you use
the Pi Camera. This code also works with USB camera connect to the Raspberry Pi.

And to see the results from the camera, you need a monitor connected
to the Raspberry Pi. It's okay if you're using SSH to access the Pi shell
(you don't need to use a keyboard connected to the Pi)—you only need a monitor
attached to the Pi to see the camera stream.


## Install the TensorFlow Lite runtime

In this project, all you need from the TensorFlow Lite API is the `Interpreter`
class. So instead of installing the large `tensorflow` package, we're using the
much smaller `tflite_runtime` package.

To install this on your Raspberry Pi, follow the instructions in the
[Python quickstart](https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python).

You can install the TFLite runtime using this script.

```
sh setup.sh
```

## Download the example files

First, clone this Git repo onto your Raspberry Pi like this:

```
git clone https://github.com/tensorflow/examples --depth 1
```

Then use our script to install a couple Python packages, and
download the TFLite model:

```
cd examples/lite/examples/image_classification/raspberry_pi

# Run this script to install the required dependencies and download the TFLite models.
sh setup.sh
```

## Run the example

```
python3 classify.py
```
*   You can optionally specify the `model` parameter to set the TensorFlow Lite
    model to be used:
    *   The default value is `efficientnet_lite0.tflite`
    *   TensorFlow Lite image classification models **with metadatafrom**
    (including models from [TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1)
    or models trained with TensorFlow Lite Model Maker are supported.)
*   You can optionally specify the `maxResults` parameter to limit the list of
    classification results:
    *   Supported value: A positive integer.
    *   Default value: `3`.
*   Example usage:

```
python3 classify.py \
  --model efficientnet_lite0.tflite \
  --maxResults 5
```

## Speed up the inferencing time (optional)

If you want to significantly speed up the inference time, you can attach an
ML accelerator such as the [Coral USB Accelerator](
https://coral.withgoogle.com/products/accelerator)—a USB accessory that adds
the [Edge TPU ML accelerator](https://coral.withgoogle.com/docs/edgetpu/faq/)
to any Linux-based system.

If you have a Coral USB Accelerator, you can run the sample with it enabled:

1.  First, be sure you have completed the [USB Accelerator setup instructions](
    https://coral.withgoogle.com/docs/accelerator/get-started/).

2.  Run the image classification script using the Edge TPU TFLite model and
    enable the Edge TPU option.

```
python3 classify.py \
  --model efficientnet_lite0_edgetpu.tflite \
  --enableEdgeTPU
```

You should see significantly faster inference speeds.

For more information about creating and running TensorFlow Lite models with
Coral devices, read [TensorFlow models on the Edge TPU](
https://coral.withgoogle.com/docs/edgetpu/models-intro/).

For more information about executing inferences with TensorFlow Lite, read
[TensorFlow Lite inference](https://www.tensorflow.org/lite/guide/inference).
