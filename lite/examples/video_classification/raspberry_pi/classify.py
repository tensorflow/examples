# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run video classification."""

import argparse
import sys
import time

import cv2

from video_classifier import VideoClassifier
from video_classifier import VideoClassifierOptions

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_MODEL_FPS = 5  # Ensure the input images are fed to the model at this fps.
_MODEL_FPS_ERROR_RANGE = 0.1  # Acceptable error range in fps.


def run(model: str, label: str, max_results: int, num_threads: int,
        camera_id: int, width: int, height: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the TFLite video classification model.
      label: Name of the video classification label.
      max_results: Max of classification results.
      num_threads: Number of CPU threads to run the model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """
  # Initialize the video classification model
  options = VideoClassifierOptions(
      num_threads=num_threads, max_results=max_results)
  classifier = VideoClassifier(model, label, options)

  # Variables to calculate FPS
  counter, fps, last_inference_start_time, time_per_infer = 0, 0, 0, 0
  categories = []

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    counter += 1

    # Mirror the image
    image = cv2.flip(image, 1)

    # Ensure that frames are feed to the model at {_MODEL_FPS} frames per second
    # as required in the model specs.
    current_frame_start_time = time.time()
    diff = current_frame_start_time - last_inference_start_time
    if diff * _MODEL_FPS >= (1 - _MODEL_FPS_ERROR_RANGE):
      # Store the time when inference starts.
      last_inference_start_time = current_frame_start_time

      # Calculate the inference FPS
      fps = 1.0 / diff

      # Convert the frame to RGB as required by the TFLite model.
      frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Feed the frame to the video classification model.
      categories = classifier.classify(frame_rgb)

      # Calculate time required per inference.
      time_per_infer = time.time() - current_frame_start_time

    # Notes: Frames that aren't fed to the model are still displayed to make the
    # video look smooth. We'll show classification results from the latest
    # classification run on the screen.
    # Show the FPS .
    fps_text = 'Current FPS = {0:.1f}. Expect: {1}'.format(fps, _MODEL_FPS)
    text_location = (_LEFT_MARGIN, _ROW_SIZE)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    # Show the time per inference.
    time_per_infer_text = 'Time per inference: {0}ms'.format(
        int(time_per_infer * 1000))
    text_location = (_LEFT_MARGIN, _ROW_SIZE * 2)
    cv2.putText(image, time_per_infer_text, text_location,
                cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR,
                _FONT_THICKNESS)

    # Show classification results on the image.
    for idx, category in enumerate(categories):
      class_name = category.label
      probability = round(category.score, 2)
      result_text = class_name + ' (' + str(probability) + ')'
      # Skip the first 2 lines occupied by the fps and time per inference.
      text_location = (_LEFT_MARGIN, (idx + 3) * _ROW_SIZE)
      cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                  _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('video_classification', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of video classification model.',
      required=False,
      default='movinet_a0_int8.tflite')
  parser.add_argument(
      '--label',
      help='Name of video classification label.',
      required=False,
      default='kinetics600_label_map.txt')
  parser.add_argument(
      '--maxResults',
      help='Max of classification results.',
      required=False,
      default=3)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      default=4)
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, args.label, int(args.maxResults), int(args.numThreads),
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()
