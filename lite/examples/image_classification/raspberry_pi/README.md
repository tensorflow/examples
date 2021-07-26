# TensorFlow Lite Python classification example with Pi Camera

This example uses [TensorFlow Lite](https://tensorflow.org/lite) with Python
on a Raspberry Pi to perform real-time image classification using images
streamed from the Pi Camera.

Although the TensorFlow model and nearly all the code in here can work with
other hardware, the code in `classify_picamera.py` uses the [`picamera`](
https://picamera.readthedocs.io/en/latest/) API to capture images from the Pi
Camera. So you can modify those parts of the code if you want to use a different
camera input.


## Set up your hardware

Before you begin, you need to [set up your Raspberry Pi](
https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up) with
Raspberry Pi OS (preferably updated to Buster).

You also need to [connect and configure the Pi Camera](
https://www.raspberrypi.org/documentation/configuration/camera.md).

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
Return here after you perform the `apt-get install` command.


## Download the example files

First, clone this Git repo onto your Raspberry Pi like this:

```shell
git clone https://github.com/tensorflow/examples --depth 1
```

Then use our script to install a couple Python packages, and
download the MobileNet model and labels file:

```shell
cd examples/lite/examples/image_classification/raspberry_pi

# The script takes an argument specifying where you want to save the model files
bash download.sh /tmp
```


## Run the example

```shell
python3 classify_picamera.py \
  --model /tmp/mobilenet_v1_1.0_224_quant.tflite \
  --labels /tmp/labels_mobilenet_quant_v1_224.txt
```

You should see the camera feed appear on the monitor attached to your Raspberry
Pi. Put some objects in front of the camera, like a coffee mug or keyboard, and
you'll see the predictions printed. It also prints the amount of time it took
to perform each inference in milliseconds.

For more information about executing inferences with TensorFlow Lite, read
[TensorFlow Lite inference](https://www.tensorflow.org/lite/guide/inference).


## Speed up the inferencing time (optional)

If you want to significantly speed up the inference time, you can attach an
ML accelerator such as the [Coral USB Accelerator](
https://coral.withgoogle.com/products/accelerator)—a USB accessory that adds
the [Edge TPU ML accelerator](https://coral.withgoogle.com/docs/edgetpu/faq/)
to any Linux-based system.

If you have a Coral USB Accelerator, follow these additional steps to
delegate model execution to the Edge TPU processor:

1.  First, be sure you have completed the [USB Accelerator setup instructions](
    https://coral.withgoogle.com/docs/accelerator/get-started/#set-up-on-linux-or-raspberry-pi).

2.  Now open the `classify_picamera.py` file and add the following import at
    the top:

    ```python
    from tflite_runtime.interpreter import load_delegate
    ```

    And then find the line that initializes the `Interpreter`, which looks like
    this:

    ```python
    interpreter = Interpreter(args.model)
    ```

    And change it to specify the Edge TPU delegate:

    ```python
    interpreter = Interpreter(args.model,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    ```

    The `libedgetpu.so.1.0` file is provided by the Edge TPU library you
    installed during the USB Accelerator setup in step 1.

3.  Finally, you need a version of the model that's compiled for the Edge TPU.

    Normally, you need to use use the [Edge TPU Compiler](
    https://coral.withgoogle.com/docs/edgetpu/compiler/) to compile your
    `.tflite` file. But the compiler tool isn't compatible with Raspberry
    Pi, so we included a pre-compiled version of the model in the `download.sh`
    script above.

    So you already have the compiled model you need:
    `mobilenet_v1_1.0_224_quant_edgetpu.tflite`.

Now you're ready to execute the TensorFlow Lite model on the Edge TPU. Just run
`classify_picamera.py` again, but be sure you specify the model that's compiled
for the Edge TPU (it uses the same labels file as before):

```shell
python3 classify_picamera.py \
  --model /tmp/mobilenet_v1_1.0_224_quant_edgetpu.tflite \
  --labels /tmp/labels_mobilenet_quant_v1_224.txt
```

You should see significantly faster inference speeds.

For more information about creating and running TensorFlow Lite models with
Coral devices, read [TensorFlow models on the Edge TPU](
https://coral.withgoogle.com/docs/edgetpu/models-intro/).
