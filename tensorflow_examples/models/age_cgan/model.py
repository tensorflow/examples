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
"""Age-cGAN Model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.')

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Upsampling2D, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, add, ZeroPadding2D, LeakyReLU

DATASET_PATH = ''
IMG_WIDTH = 128
IMG_HEIGHT = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, shape):
        self.scale = self.add_weight(name='scale', shape=shape[-1:], initializer=tf.keras.initializers.RandomNormal(0.0, 0.002), trainable=True)
        self.offset = self.add_weight(name='offset', shape=shape[-1:], initializer='zeros', trainable=True)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axis=1, keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        x = (inputs - mean) * inv
        return self.scale * x + self.offset

def load_data(path):
    pass

def build_generator():
    """An Autoencoder network.
    """
    # Convolution Block
    input_layer1 = Input(shape=(128, 128, 3))
    x = Conv2D(32, (7,7), strides=1, padding='same', use_bias=False)(input_layer1)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3,3), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3,3), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
      
    # Residual block.
    x1 = Conv2D(128, (3,3), strides=1, padding='same', use_bias=False)(x)
    x1 = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(x1)
    x1 = Conv2D(128, (3,3), strides=1, padding='same', use_bias=False)(x1)
    x1 = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(x1)
    x = add()([x, x1])

    # Upsampling block.
    x = Conv2DTranspose(64, (3,3), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    output = Conv2D(3, (7,7), strides=1, padding='same', activation='tanh', use_bias=False)(x)

    return Model(inputs=[input_layer1], outputs=[output])

def build_discriminator():
    input_layer = Input(shape=(128, 128, 3))
    x = ZeroPadding2D(padding=(1,1))(input_layer)
    x = Conv2D(64, (4,4), strides=2, padding='valid')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = InstanceNormalization()(x)
    x = Conv2D(128, (4,4), strides=2, padding='valid')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = InstanceNormalization()(x)
    x = Conv2D(256, (4,4), strides=2, padding='valid')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = InstanceNormalization()(x)
    x = Conv2D(512, (4,4), strides=2, padding='valid')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    output = Conv2D(1, (4,4), strides=1, padding='valid', activation='sigmoid')(x)

    return Model(inputs=[input_layer], outputs=[output])


def run_main(argv):
    del argv
    kwargs = {'path' : DATASET_PATH}
    main(**kwargs)

def main(path):
    pass

if __name__ == '__main__':
    app.run(run_main)