# TensorFlow Lite Python video classification example with Raspberry Pi.

This example uses [TensorFlow Lite](https://tensorflow.org/lite) with Python on
a Raspberry Pi to perform real-time video classification using images streamed
from the camera.

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


## Setup Environment

To ensure the TensorFlow Lite examples run smoothly on Raspberry Pi OS based on Debian Bookworm (2024 release), setting up a Python virtual environment is crucial. This setup guarantees compatibility with Python 3.9 and effectively manages the dependencies for your examples.

### Install Python 3.9.0

Follow these steps to install Python from source:

1. **Install Build Dependencies**: Begin by updating your system and installing the required packages:
   ```bash
   sudo apt update
   sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
   ```

2. **Compile and Install Python**: Download, extract, and install Python:
   ```bash
   wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
   tar xf Python-3.9.0.tar.xz
   cd Python-3.9.0
   ./configure --enable-optimizations --prefix=/usr
   make -j $(nproc)
   sudo make altinstall
   ```

3. **Verify Python Installation**: Ensure Python 3.9.0 is installed successfully:
   ```bash
   python3.9 --version
   ```

4. **Change Directory**: Move out of the Python source directory:
   ```bash
   cd ..
   ```

### Optional: Cleanup After Installation

To optimize disk space after the Python installation:

1. **Remove Python Source Files**: Next, use `sudo` to  remove the downloaded archive and the extracted Python source directory:
   ```bash
   sudo rm -rf Python-3.9.0.tar.xz Python-3.9.0
   ```

2. **Remove Build Dependencies**: Optionally, uninstall the build dependencies if they are no longer necessary:
   ```bash
   sudo apt remove --purge -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
   sudo apt autoremove -y
   ```

### Virtual Environment Setup

Prepare and activate a virtual environment for the TensorFlow Lite examples:

1. **Create the Environment**: 
   ```bash
   python3.9 -m venv /usr/local/venvs/tflite
   ```

2. **Activate the Environment**: Always reactivate the virtual environment in new sessions:
   ```bash
   source /usr/local/venvs/tflite/bin/activate
   ```

**Note**: Remember to reactivate the `tflite` environment with `/usr/local/venvs/tflite/bin/activate` each time you work on the TensorFlow Lite examples.


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

Then use our script to install a couple Python packages:

```
cd examples/lite/examples/video_classification/raspberry_pi

# The script install the required dependencies and download the TFLite models.
sh setup.sh
```

## Run the example

```
python3 classify.py
```

*   You can optionally specify the `model` parameter to set the TensorFlow Lite
    model to be used:
    *   The default value is `movinet_a0_int8.tflite`
    *   This sample currently uses MoviNet-A0, but it supports all TensorFlow
        Lite
        [MoviNet video classification](https://tfhub.dev/s?deployment-format=lite&q=movinet)
        models that are available on TensorFlow Hub. You can use the larger
        variant of MoviNet if you need higher accuracy.
*   You can optionally specify the `maxResults` parameter to limit the list of
    classification results:
    *   Supported value: A positive integer.
    *   Default value: `3`.
*   Example usage: `python3 classify.py \ --model movinet_a0_int8.tflite \
    --maxResults 5`

For more information about executing inferences with TensorFlow Lite, read
[TensorFlow Lite inference](https://www.tensorflow.org/lite/guide/inference).
