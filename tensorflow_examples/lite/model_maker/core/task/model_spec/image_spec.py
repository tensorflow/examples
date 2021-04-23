# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Image model specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core.api import mm_export
from tensorflow_examples.lite.model_maker.core.task.model_spec import util


@mm_export('image_classifier.ImageSpec')
class ImageModelSpec(object):
  """A specification of image model."""

  mean_rgb = [0.0]
  stddev_rgb = [255.0]

  def __init__(self,
               uri,
               compat_tf_versions=None,
               input_image_shape=None,
               name=''):
    """Initializes a new instance of the `ImageModelSpec` class.

    Args:
      uri: str, URI to the pretrained model.
      compat_tf_versions: list of int, compatible TF versions.
      input_image_shape: list of int, input image shape. Default: [224, 224].
      name: str, model spec name.
    """
    self.uri = uri
    self.compat_tf_versions = compat.get_compat_tf_versions(compat_tf_versions)
    self.name = name

    if input_image_shape is None:
      input_image_shape = [224, 224]
    self.input_image_shape = input_image_shape


mobilenet_v2_spec = functools.partial(
    ImageModelSpec,
    uri='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
    compat_tf_versions=2,
    name='mobilenet_v2')
mobilenet_v2_spec.__doc__ = util.wrap_doc(ImageModelSpec,
                                          'Creates MobileNet v2 model spec.')
mm_export('image_classifier.MobileNetV2Spec').export_constant(
    __name__, 'mobilenet_v2_spec')

resnet_50_spec = functools.partial(
    ImageModelSpec,
    uri='https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',
    compat_tf_versions=2,
    name='resnet_50')
resnet_50_spec.__doc__ = util.wrap_doc(ImageModelSpec,
                                       'Creates ResNet 50 model spec.')
mm_export('image_classifier.Resnet50Spec').export_constant(
    __name__, 'resnet_50_spec')

efficientnet_lite0_spec = functools.partial(
    ImageModelSpec,
    uri='https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2',
    compat_tf_versions=[1, 2],
    name='efficientnet_lite0')
efficientnet_lite0_spec.__doc__ = util.wrap_doc(
    ImageModelSpec, 'Creates EfficientNet-Lite0 model spec.')
mm_export('image_classifier.EfficientNetLite0Spec').export_constant(
    __name__, 'efficientnet_lite0_spec')

efficientnet_lite1_spec = functools.partial(
    ImageModelSpec,
    uri='https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[240, 240],
    name='efficientnet_lite1')
efficientnet_lite1_spec.__doc__ = util.wrap_doc(
    ImageModelSpec, 'Creates EfficientNet-Lite1 model spec.')
mm_export('image_classifier.EfficientNetLite1Spec').export_constant(
    __name__, 'efficientnet_lite1_spec')

efficientnet_lite2_spec = functools.partial(
    ImageModelSpec,
    uri='https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[260, 260],
    name='efficientnet_lite2')
efficientnet_lite2_spec.__doc__ = util.wrap_doc(
    ImageModelSpec, 'Creates EfficientNet-Lite2 model spec.')
mm_export('image_classifier.EfficientNetLite2Spec').export_constant(
    __name__, 'efficientnet_lite2_spec')

efficientnet_lite3_spec = functools.partial(
    ImageModelSpec,
    uri='https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[280, 280],
    name='efficientnet_lite3')
efficientnet_lite3_spec.__doc__ = util.wrap_doc(
    ImageModelSpec, 'Creates EfficientNet-Lite3 model spec.')
mm_export('image_classifier.EfficientNetLite3Spec').export_constant(
    __name__, 'efficientnet_lite3_spec')

efficientnet_lite4_spec = functools.partial(
    ImageModelSpec,
    uri='https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[300, 300],
    name='efficientnet_lite4')
efficientnet_lite4_spec.__doc__ = util.wrap_doc(
    ImageModelSpec, 'Creates EfficientNet-Lite4 model spec.')
mm_export('image_classifier.EfficientNetLite4Spec').export_constant(
    __name__, 'efficientnet_lite4_spec')
