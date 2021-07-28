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
    *   Use values: `posenet`, `movenet_lightning`, `movenet_thunder`.
    *   The default value is `movenet_lightning`.

```
python3 pose_estimation.py --model_name movenet_thunder
```

## Run the pose classification sample

*   Use this command to run the pose estimation sample using the default
    `movenet_lightning` pose estimation model and the `yoga_classifier.tflite`
    pose classification model.

```
python3 pose_estimation.py \
    --classifier yoga_classifier
    --label_file yoga_labels.txt
```

## Customization options

*   Here is the full list of parameters supported by the sample:
```python3 pose_classification.py \
      --model model_name \
      --classifier classification_model_name`
```
*   `model`: Name of the TFLite pose estimation model to be used.
    *   One of these values: `posenet`, `movenet_lightning`, `movenet_thunder`.
    *   Default value is `movenet_lightning`.
*   `classifier`: Name of the TFLite pose classification model to be used.
    *   Default value is empty.
    *   If no classification model specified, the sample will only run the pose
        estimation step.
*   `camera_id`: Specify the camera for OpenCV to capture images from.
    *   Default value is `0`.
*   `frameWidth`, `frameHeight`: Resolution of the image to be captured from the
    camera.
    *   Default value is `(640, 480)`.
