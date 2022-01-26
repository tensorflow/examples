# TensorFlow Lite Python audio classification example with Raspberry Pi.

This example uses [TensorFlow Lite](https://tensorflow.org/lite) with Python on
a Raspberry Pi to perform real-time audio classification using audio streamed
from the microphone.

At the end of this page, there are extra steps to accelerate the example using
the Coral USB Accelerator to increase inference speed.

## Set up your hardware

Before you begin, you need to
[set up your Raspberry Pi](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up)
with Raspberry Pi OS (preferably updated to Buster).

Raspberry Pi doesn't have a microphone integrated on its board, so you need to
plug in a USB microphone to record audio.

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
git clone https://github.com/tensorflow/examples.git --depth 1
```

Then use our script to install a couple Python packages, and download the TFLite
model:

```
cd lite/examples/sound_classification/raspberry_pi

# Run this script to install the required dependencies and download the TFLite models.
sh setup.sh
```

## Run the example

```
python3 classify.py
```

*   You can optionally specify the `model` parameter to set the TensorFlow Lite
    model to be used:
    *   The default value is `yamnet.tflite`
*   You can optionally specify the `maxResults` parameter to limit the list of
    classification results:
    *   Supported value: A positive integer.
    *   Default value: `5`.
*   Example usage:

```
python3 classify.py \
  --model yamnet.tflite \
  --maxResults 5
```

## Speed up the inference time (optional)

If you want to speed up the inference time, you can attach an ML accelerator
such as the
[Coral USB Accelerator](https://coral.withgoogle.com/products/accelerator)—a USB
accessory that adds the
[Edge TPU ML accelerator](https://coral.withgoogle.com/docs/edgetpu/faq/) to any
Linux-based system.

If you have a Coral USB Accelerator, you can run the sample with it enabled:

1.  First, be sure you have completed the
    [USB Accelerator setup instructions](https://coral.withgoogle.com/docs/accelerator/get-started/).

2.  Run the audio classification script using the Edge TPU TFLite model and
    enable the Edge TPU option.

```
python3 classify.py \
  --model yamnet_edgetpu.tflite \
  --enableEdgeTPU
```

For more information about creating and running TensorFlow Lite models with
Coral devices, read
[TensorFlow models on the Edge TPU](https://coral.withgoogle.com/docs/edgetpu/models-intro/).

For more information about executing inferences with TensorFlow Lite, read
[TensorFlow Lite inference](https://www.tensorflow.org/lite/guide/inference).
