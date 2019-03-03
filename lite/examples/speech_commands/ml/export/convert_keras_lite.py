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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

keras_model = "../conv_1d_time_stacked_model/ep-084-vl-0.2595.hdf5"
input_arrays = ["the_input"]
output_arrays = ["the_output"]

converter = tf.contrib.lite.TocoConverter
converter = converter.from_keras_model_file(keras_model, input_arrays,
                                            output_arrays)
tflite_model = converter.convert()
open("converted_speed_keras_model.tflite", "wb").write(tflite_model)
