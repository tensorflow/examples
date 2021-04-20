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
    self.uri = uri
    self.compat_tf_versions = compat.get_compat_tf_versions(compat_tf_versions)
    self.name = name

    if input_image_shape is None:
      input_image_shape = [224, 224]
    self.input_image_shape = input_image_shape


@mm_export('image_classifier.MobilenetV2Spec')
def mobilenet_v2_spec(**kwargs):
  args = util.dict_with_default(
      default_dict=dict(
          uri='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
          compat_tf_versions=2,
          name='mobilenet_v2'),
      **kwargs)
  return ImageModelSpec(**args)


@mm_export('image_classifier.Resnet50Spec')
def resnet_50_spec(**kwargs):
  args = util.dict_with_default(
      default_dict=dict(
          uri='https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',
          compat_tf_versions=2,
          name='resnet_50'),
      **kwargs)
  return ImageModelSpec(**args)


@mm_export('image_classifier.EfficientNetLite0Spec')
def efficientnet_lite0_spec(**kwargs):
  """Model specification for EfficientNet-Lite0."""
  args = util.dict_with_default(
      default_dict=dict(
          uri='https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2',
          compat_tf_versions=[1, 2],
          name='efficientnet_lite0'),
      **kwargs)
  return ImageModelSpec(**args)


@mm_export('image_classifier.EfficientNetLite1Spec')
def efficientnet_lite1_spec(**kwargs):
  """Model specification for EfficientNet-Lite1."""
  args = util.dict_with_default(
      default_dict=dict(
          uri='https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2',
          compat_tf_versions=[1, 2],
          input_image_shape=[240, 240],
          name='efficientnet_lite1'),
      **kwargs)
  return ImageModelSpec(**args)


@mm_export('image_classifier.EfficientNetLite2Spec')
def efficientnet_lite2_spec(**kwargs):
  """Model specification for EfficientNet-Lite2."""
  args = util.dict_with_default(
      default_dict=dict(
          uri='https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2',
          compat_tf_versions=[1, 2],
          input_image_shape=[260, 260],
          name='efficientnet_lite2'),
      **kwargs)
  args.update(**kwargs)
  return ImageModelSpec(**args)


@mm_export('image_classifier.EfficientNetLit3Spec')
def efficientnet_lite3_spec(**kwargs):
  """Model specification for EfficientNet-Lite3."""
  args = util.dict_with_default(
      default_dict=dict(
          uri='https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2',
          compat_tf_versions=[1, 2],
          input_image_shape=[280, 280],
          name='efficientnet_lite3'),
      **kwargs)
  return ImageModelSpec(**args)


@mm_export('image_classifier.EfficientNetLite4Spec')
def efficientnet_lite4_spec(**kwargs):
  """Model specification for EfficientNet-Lite4."""
  args = util.dict_with_default(
      default_dict=dict(
          uri='https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2',
          compat_tf_versions=[1, 2],
          input_image_shape=[300, 300],
          name='efficientnet_lite4'),
      **kwargs)
  return ImageModelSpec(**args)
