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
"""TensorFlow 2.0 implementation of vanilla autoencoder"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, units, latent_dim):
        super(Encoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(784, ))
        self.hidden_layer = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu)

    def call(self, input_features):
        input_features = self.input_layer(input_features)
        activations = self.hidden_layer(input_features)
        logits = self.output_layer(activations)
        return logits


class Decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, original_dim):
        super(Decoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(latent_dim, ))
        self.hidden_layer = tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.sigmoid)

    def call(self, input_features):
        input_features = self.input_layer(input_features)
        activations = self.hidden_layer(input_features)
        reconstructed = self.output_layer(activations)
        return reconstructed


class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim, original_dim, units):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(units=units, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, original_dim=original_dim)
        self.train_losses = []

    @tf.function
    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

    @tf.function
    def loss_fn(self, reconstructed, original):
        return tf.reduce_mean(tf.square(tf.subtract(reconstructed, original)))

    def train(self, dataset, epochs):
        optimizer = tf.optimizers.Adam(learning_rate=3e-4)

        for epoch in range(epochs):
            for step, batch_features in enumerate(dataset):
                with tf.GradientTape() as tape:
                    reconstructed = self.call(batch_features)
                    train_loss = self.loss_fn(reconstructed, batch_features)
                gradients = tape.gradient(train_loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                self.train_losses.append(train_loss)

            print('Epoch {}/{} : mean loss = {}'.format(epoch + 1, epochs, tf.reduce_mean(self.train_losses)))
