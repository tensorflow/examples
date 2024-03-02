# Pose estimation and classification with TensorFlow Lite

See this blog post (TBD) for a full guide on doing pose estimation and
classification using TensorFlow Lite.
*   Pose estimation: Detect keypoints, such as eye, ear, arm etc., from an input
    image.
    *   Input: An image
    *   Output: A list of keypoint coordinates and confidence score.
*   Pose classificaiton: Classify a human pose into predefined classes, such as
    different yoga poses. Pose classification internally use pose estimation to
    detect the keypoints, and use the keypoints to classify the pose.
    *   Input: An image
    *   Output: A list of predefined classes and their confidence score.

This sample can run on Raspberry Pi or any computer that has a camera. It uses
OpenCV to capture images from the camera and TensorFlow Lite to run inference on
the input image.


## Setup Environment

To ensure the TensorFlow Lite examples run smoothly on Raspberry Pi OS based on Debian Bookworm (2024 release), setting up a Python virtual environment is crucial. This setup guarantees compatibility with Python 3.9 and effectively manages the dependencies for your examples.

### Install Python 3.9

Follow these steps to install Python 3.9 from source:

1. **Install Build Dependencies**: Begin by updating your system and installing the required packages:
   ```bash
   sudo apt update
   sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
   ```

2. **Compile and Install Python 3.9**: Download, extract, and install Python 3.9:
   ```bash
   wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
   tar xf Python-3.9.0.tar.xz
   cd Python-3.9.0
   ./configure --enable-optimizations --prefix=/usr
   make -j $(nproc)
   sudo make altinstall
   ```

3. **Verify Python Installation**: Ensure Python 3.9 is installed successfully:
   ```bash
   python3.9 --version
   ```

### Optional: Cleanup After Installation

To optimize disk space after the Python installation:

1. **Remove Python Source Files**: Eliminate the no-longer-needed source files:
   ```bash
   cd ..
   rm -rf Python-3.9.0.tar.xz Python-3.9.0
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
   python3.9 -m venv tflite
   ```

2. **Activate the Environment**: Always reactivate the virtual environment in new sessions:
   ```bash
   source tflite/bin/activate
   ```

**Note**: Remember to reactivate the `tflite` environment with `source tflite/bin/activate` each time you work on the TensorFlow Lite examples.


## Install the dependencies

*   Run this script to install the Python dependencies, and download the TFLite
    models. `sh setup.sh`

## Run the pose estimation sample

*   Use this command to run the pose estimation sample using the default
    `movenet_lightning` model.

```
python3 pose_estimation.py
```

*   You can optionally specify the `model_name` parameter to try other pose
    estimation models:
    *   Use values:
        * Single-pose: `posenet`, `movenet_lightning`, `movenet_thunder`
        * Multi-poses: `movenet_multipose`
    *   The default value is `movenet_lightning`.

```
python3 pose_estimation.py --model_name movenet_thunder
```

## Run the pose classification sample

*   Use this command to run the pose estimation sample using the default
    `movenet_lightning` pose estimation model and the `classifier.tflite` yoga
    pose classification model.

```
python3 pose_estimation.py \
    --classifier classifier
    --label_file labels.txt
```

*   If you want to train a custom pose classification model, check out
    [this tutorial](https://www.tensorflow.org/lite/tutorials/pose_classification).

## Customization options

*  Here is the full list of parameters supported by the sample:
```python3 pose_classification.py```
  *   `model`: Name of the TFLite pose estimation model to be used.
    *   One of these values: `posenet`, `movenet_lightning`, `movenet_thunder`, `movenet_multipose`
    *   Default value is `movenet_lightning`.
  *   `tracker`: Type of tracker to track poses across frames.
    *   One of these values: `bounding_box`, `keypoint`
    *   Only supported in multi-poses models.
    *   Default value is `bounding_box`.
  *   `classifier`: Name of the TFLite pose classification model to be used.
    *   Default value is empty.
    *   If no classification model specified, the sample will only run the pose
        estimation step.
  *   `camera_id`: Specify the camera for OpenCV to capture images from.
    *   Default value is `0`.
  *   `frameWidth`, `frameHeight`: Resolution of the image to be captured from
      the camera.
    *   Default value is `(640, 480)`.

## Visualize pose estimation result of test data

*  Run this script to visualize the pose estimation on test data

```python3 visualizer.py```
