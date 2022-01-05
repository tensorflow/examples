# TensorFlow Lite Python object detection example with Raspberry Pi

This example uses [TensorFlow Lite](https://tensorflow.org/lite) with Python on
a Raspberry Pi to perform real-time object detection using images streamed from
the Pi Camera. It draws a bounding box around each detected object in the camera
preview (when the object score is above a given threshold).

At the end of this page, there are extra steps to accelerate the example using
the Coral USB Accelerator to increase inference speed.

## Set up your hardware

Before you begin, you need to
[set up your Raspberry Pi](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up)
with Raspberry Pi OS (preferably updated to Buster).

You also need to
[connect and configure the Pi Camera](https://www.raspberrypi.org/documentation/configuration/camera.md)
if you use the Pi Camera. This code also works with USB camera connect to the
Raspberry Pi.

And to see the results from the camera, you need a monitor connected to the
Raspberry Pi. It's okay if you're using SSH to access the Pi shell (you don't
need to use a keyboard connected to the Pi)—you only need a monitor attached to
the Pi to see the camera stream.

## Download the example files

First, clone this Git repo onto your Raspberry Pi like this:

```
git clone https://github.com/tensorflow/examples --depth 1
```

Then use our script to install a couple Python packages, and download the
EfficientDet-Lite model:

```
cd examples/lite/examples/object_detection/raspberry_pi

# The script install the required dependencies and download the TFLite models.
sh setup.sh
```

In this project, all you need from the TensorFlow Lite API is the `Interpreter`
class. So instead of installing the large `tensorflow` package, we're using the
much smaller `tflite_runtime` package. The setup scripts automatically install
the TensorFlow Lite runtime.

## Run the example

```
python3 detect.py \
  --model efficientdet_lite0.tflite
```

You should see the camera feed appear on the monitor attached to your Raspberry
Pi. Put some objects in front of the camera, like a coffee mug or keyboard, and
you'll see boxes drawn around those that the model recognizes, including the
label and score for each. It also prints the number of frames per second (FPS)
at the top-left corner of the screen. As the pipeline contains some processes
other than model inference, including visualizing the detection results, you can
expect a higher FPS if your inference pipeline runs in headless mode without
visualization.

For more information about executing inferences with TensorFlow Lite, read
[TensorFlow Lite inference](https://www.tensorflow.org/lite/guide/inference).

## Speed up model inference (optional)

If you want to significantly speed up the inference time, you can attach an
[Coral USB Accelerator](https://coral.withgoogle.com/products/accelerator)—a USB
accessory that adds the
[Edge TPU ML accelerator](https://coral.withgoogle.com/docs/edgetpu/faq/) to any
Linux-based system.

If you have a Coral USB Accelerator, you can run the sample with it enabled:

1.  First, be sure you have completed the
    [USB Accelerator setup instructions](https://coral.withgoogle.com/docs/accelerator/get-started/).

2.  Run the object detection script using the EdgeTPU TFLite model and enable
    the EdgeTPU option. Be noted that the EdgeTPU requires a specific TFLite
    model that is different from the one used above.

```
python3 detect.py \
  --enableEdgeTPU
  --model efficientdet_lite0_edgetpu.tflite
```

You should see significantly faster inference speeds.

For more information about creating and running TensorFlow Lite models with
Coral devices, read
[TensorFlow models on the Edge TPU](https://coral.withgoogle.com/docs/edgetpu/models-intro/).
