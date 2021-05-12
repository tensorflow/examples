# Copyright 2021 Google Research. All Rights Reserved.
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
# ==============================================================================
r"""Run TF Lite model."""
from absl import app
from absl import flags

from PIL import Image
import tensorflow as tf

from tensorflow_examples.lite.model_maker.third_party.efficientdet import inference

FLAGS = flags.FLAGS


def define_flags():
  """Define flags."""
  flags.DEFINE_string('tflite_path', None, 'Path of tflite file.')
  flags.DEFINE_string('sample_image', None, 'Sample image path')
  flags.DEFINE_string('output_image', None, 'Output image path')
  flags.DEFINE_string('image_size', '512x512', 'Image size "WxH".')


def load_image(image_path, image_size):
  """Loads an image, and returns numpy.ndarray.

  Args:
    image_path: str, path to image.
    image_size: list of int, representing [width, height].

  Returns:
    image_batch: numpy.ndarray of shape [1, H, W, C].
  """
  input_data = tf.io.gfile.GFile(image_path, 'rb').read()
  image = tf.io.decode_image(input_data, channels=3, dtype=tf.uint8)
  image = tf.image.resize(
      image, image_size, method='bilinear', antialias=True)
  return tf.expand_dims(tf.cast(image, tf.uint8), 0).numpy()


def save_visualized_image(image, prediction, output_path):
  """Saves the visualized image with prediction.

  Args:
    image: numpy.ndarray of shape [H, W, C].
    prediction: numpy.ndarray of shape [num_predictions, 7].
    output_path: str, output image path.
  """
  output_image = inference.visualize_image_prediction(
      image,
      prediction,
      label_map='coco')
  Image.fromarray(output_image).save(output_path)


class TFLiteRunner:
  """Wrapper to run TFLite model."""

  def __init__(self, model_path):
    """Init.

    Args:
      model_path: str, path to tflite model.
    """
    self.interpreter = tf.lite.Interpreter(model_path=model_path)
    self.interpreter.allocate_tensors()
    self.input_index = self.interpreter.get_input_details()[0]['index']
    self.output_index = self.interpreter.get_output_details()[0]['index']

  def run(self, image):
    """Run inference on a single images.

    Args:
      image: numpy.ndarray of shape [1, H, W, C].

    Returns:
      prediction: numpy.ndarray of shape [1, num_detections, 7].
    """
    self.interpreter.set_tensor(self.input_index, image)
    self.interpreter.invoke()
    return self.interpreter.get_tensor(self.output_index)


def main(_):
  image_size = [int(dim) for dim in FLAGS.image_size.split('x')]
  image = load_image(FLAGS.sample_image, image_size)

  runner = TFLiteRunner(FLAGS.tflite_path)
  prediction = runner.run(image)

  save_visualized_image(image[0], prediction[0], FLAGS.output_image)


if __name__ == '__main__':
  define_flags()
  app.run(main)
