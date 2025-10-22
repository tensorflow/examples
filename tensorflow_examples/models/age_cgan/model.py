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

Dataset Citation:
@article{Rothe-IJCV-2016,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {Deep expectation of real and apparent age from a single image without facial landmarks},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2016},
  month = {July},
}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.')

from scipy.io import loadmat
from datetime import datetime
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, UpSampling2D, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, add, ZeroPadding2D, LeakyReLU
from tensorflow.keras.layers import Lambda, Reshape, Flatten, concatenate, Dropout
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

DATASET_PATH = "./wiki_crop/"

def compute_age(photo_date, dob):
    """Calculates the age from the dob and the date of photo taken.
    """
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    if birth.month < 7:
        return photo_date - birth.year
    else:
        return photo_date - birth.year - 1

def load_data(path):
    """Loads the image paths and the calculated ages for each photo. 
    """
    metadata = loadmat(os.path.join(path, "wiki.mat"))
    paths = metadata['wiki'][0, 0]['full_path'][0]
    dob = metadata['wiki'][0, 0]['dob'][0]
    photo_date = metadata['wiki'][0, 0]['photo_taken'][0]
    calculated_age = [compute_age(photo_date[i], dob[i]) for i in range(len(dob))]

    images = []
    ages_list = []

    for i, image_path in enumerate(paths):
        images.append(image_path[0])
        ages_list.append(calculated_age[i])

    return images, ages_list

def make_encoder():
    """Builds the Encoder network.
    """
    input_layer = Input(shape=(64, 64, 3))
    x = Conv2D(32, (5,5), strides=2, padding='same', kernel_initializer='he_uniform')(input_layer)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, (5,5), strides=2, padding='same', kernel_initializer='he_uniform')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (5,5), strides=2, padding='same', kernel_initializer='he_uniform')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, (5,5), strides=2, padding='same', kernel_initializer='he_uniform')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(4096)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(100)(x)

    return Model(inputs=[input_layer], outputs=[x])


def make_generator():
    """Builds the Generator network.
    """
    latent_vector = Input(shape=(100,))
    conditioning_variable = Input(shape=(6,))

    x = concatenate([latent_vector, conditioning_variable])

    x = Dense(2048, input_dim=106)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)

    x = Dense(16384)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)

    x = Reshape((8, 8, 256))(x)

    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(128, (5,5), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)

    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64, (5,5), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)

    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(3, (5,5), padding='same', kernel_initializer='he_uniform')(x)
    x = Activation('tanh')(x)

    return Model(inputs=[latent_vector, conditioning_variable], outputs=[x])


def make_face_recognition(shape):
    """Builds the Face Recognition Network.
    """
    model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=shape, pooling='avg')
    image = model.input
    x = model.layers[-1].output
    outputs = Dense(128)(x)
    embedding_model = Model(inputs=[image], outputs=[outputs])

    input_layer = Input(shape=shape)
    x = embedding_model(input_layer)
    outputs = Lambda(lambda x: tf.keras.backend.l2_normalize(x, -1))(x)
    return Model(inputs=[input_layer], outputs=[outputs])

def save_image(img, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    plt.savefig(path)
    plt.close()

def expand_dims(label):
    """Increases dimensions along the specified axis.
    """
    label = tf.keras.backend.expand_dims(label, 1)
    label = tf.keras.backend.expand_dims(label, 1)
    return tf.keras.backend.tile(label, [1, 32, 32, 1])

def make_resize():
    """Returns the image resize model.
    """
    input_image = Input(shape=(64, 64, 3))
    resized_image = Lambda(lambda x: K.resize_images(x, height_factor=3, width_factor=3,
                                    data_format='channels_last'))(input_image)
    return Model(inputs=[input_image], outputs=[resized_image])

def make_discriminator():
    """Builds the Discriminator network.
    """
    image_input = Input(shape=(64, 64, 3))
    label_input = Input(shape=(6,))

    x = Conv2D(64, (3,3), strides=2, padding='same', kernel_initializer='he_uniform')(image_input)
    x = LeakyReLU(0.2)(x)

    label = Lambda(expand_dims)(label_input)
    
    x = concatenate([x, label], axis=3)
    x = Conv2D(128, (3,3), strides=2, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, (3,3), strides=2, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(512, (3,3), strides=2, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[image_input, label_input], outputs=[x])


def make_adversarial(generator, discriminator):
    """Builds the Adversarial Network.
    """
    latent_space = Input(shape=(100,))
    conditioning_variable = Input(shape=(6,))

    discriminator.trainable = False
    reconstructed = generator([latent_space, conditioning_variable])
    valid = discriminator([reconstructed, conditioning_variable])

    discriminator.trainable = True
    return Model(inputs=[latent_space, conditioning_variable], outputs=[valid])


def get_age_categories(ages):
    age_list = []
    for age in ages:
        if 0 < age <= 18:
            age_list.append(0)
        elif 18 < age <= 29:
            age_list.append(1)
        elif 29 < age <= 39:
            age_list.append(2)
        elif 39 < age <= 49:
            age_list.append(3)
        elif 49 < age <= 59:
            age_list.append(4)
        elif age >= 60:
            age_list.append(5)

    return age_list

def load_images(path, image_paths, shape):
    """Returns all the images as a variable of concatenated arrays.
    """
    image_ = None
    for i, image_path in enumerate(image_paths):
        try:
            image = tf.keras.preprocessing.image.load_img(os.path.join(path, image_path),
                        target_size=shape)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = np.expand_dims(image, axis=0)

            if image_ is None:
                image_ = image
            else:
                image_ = np.concatenate([image_, image], axis=0)

        except Exception as e:
            print(f'Error! Image {i} {e}')

    return image_

def euclidean_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def run_main(argv):
    del argv
    kwargs = {'path' : DATASET_PATH}
    main(**kwargs)  

def main(path):
    """The training of the Age-cGAN occurs in 3 steps:
        1. Training the Generator and Discriminator Networks.
        2. Initial Latent Vector Approximation (Encoder).
        3. Latent Vector Optimization (Encoder and Generator).
    """
    epochs = 250
    batch_size = 128 # Change to 1 in case of ConcatOp error
    TRAIN_GAN = True # Step 1
    TRAIN_ENCODER = True # Step 2
    TRAIN_ENC_GAN = True # Step 3
    latent_shape = 100
    image_shape = (64, 64, 3)
    fr_image_shape = (192, 192, 3)
    gen_opt = tf.keras.optimizers.Adam(lr=0.002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    dis_opt = tf.keras.optimizers.Adam(lr=0.002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    adv_opt = tf.keras.optimizers.Adam(lr=0.002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    generator = make_generator()
    generator.compile(loss='binary_crossentropy', optimizer=gen_opt)

    discriminator = make_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_opt)

    adversarial = make_adversarial(generator, discriminator)
    adversarial.compile(loss='binary_crossentropy', optimizer=adv_opt)

    images, age_list = load_data(path)
    categories = get_age_categories(age_list)
    age_categories = np.reshape(np.array(categories), [len(categories), 1])
    num_classes = len(set(categories))
    y = to_categorical(age_categories, num_classes=num_classes)

    loaded_images = load_images(path, images, (image_shape[0], image_shape[1]))

    real = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

    # Train Step 1: Train the Generator and Discriminator
    if TRAIN_GAN:
        print(f'Step 1: Training the Generator and Discriminator')
        for epoch in range(epochs):
            print(f'Epoch:{epoch}')

            num_batches = int(len(loaded_images)/batch_size)
            for i in range(num_batches):
                batch = loaded_images[i*batch_size:(i+1)*batch_size]
                batch = batch / 127.5 - 1.
                batch = batch.astype(np.float32)

                latent_vector = np.random.normal(0, 1, size=(batch_size, 100))
                y_curr = y[i*batch_size:(i+1)*batch_size]

                reconstructed = generator.predict_on_batch([latent_vector, y_curr])
                d_real = discriminator.train_on_batch([batch, y_curr], [real])
                d_recons = discriminator.train_on_batch([reconstructed, y_curr], [fake])
                d_curr = 0.5 * np.add(d_real, d_recons)

                latent_space = np.random.normal(0, 1, size=(batch_size, 100))
                conditioning_variable = np.random.randint(0, 6, batch_size).reshape(-1, 1)
                conditioning_variable = to_categorical(conditioning_variable, num_classes=6)

                g_curr = adversarial.train_on_batch([latent_space, conditioning_variable], [1]*batch_size)
            print(f'Gen_loss:{g_curr}\nDisc_loss:{d_curr}')

            if epoch % 10 == 0:
                mini_batch = loaded_images[:batch_size]
                mini_batch = mini_batch / 127.5 - 1.
                mini_batch = mini_batch.astype(np.float32)

                y_batch = y[:batch_size]
                latent_space = np.random.normal(0, 1, size=(batch_size, y_batch))

                gen = generator.predict_on_batch([latent_space, y_batch])

                for i, image in enumerate(gen[:10]):
                    save_image(image, path=f'results/image_{epoch}_{i}')

            if epoch % 25 == 0:
                generator.save_weights("generator.h5")
                discriminator.save_weights("discriminator.h5")
                adversarial.save_weights("adversarial.h5")

        generator.save_weights("generator.h5")
        discriminator.save_weights("discriminator.h5")
        adversarial.save_weights("adversarial.h5")

    # Train Step 2: Train the Encoder
    if TRAIN_ENCODER:
        print(f'Step 2: Training the Encoder - Latent Vector Approximation')
        encoder = make_encoder()
        encoder.compile(loss=euclidean_loss, optimizer=tf.keras.optimizers.Adam())

        generator.load_weights('generator.h5')
                
        latent_vector = np.random.normal(0, 1, size=(5000, 100))
        y = np.random.randint(0, 6, size=(5000,), dtype=np.int64)
        num_classes = len(set(y))
        y = np.reshape(np.array(y), [len(y), 1])
        y = to_categorical(y, num_classes=num_classes)

        for epoch in range(epochs):
            print(f'Epoch: {epoch}')

            num_batches = int(len(latent_vector.shape[0])/batch_size)
            for i in range(batch_size):

                latent_batch = latent_vector[i*batch_size:(i+1)*batch_size]
                y_batch = y[i*batch_size:(i+1)*batch_size]

                reconstructed = generator.predict_on_batch([latent_batch, y_batch])
                e_loss = encoder.train_on_batch(reconstructed, latent_batch)

            print(f'Encoder_loss: {e_loss}')

            if epoch % 25 == 0:
                encoder.save_weights('encoder.h5')

        encoder.save_weights('encoder.h5')

    # Train Step 3: Train the Generator and Encoder and Generator
    if TRAIN_ENC_GAN:
        print(f'Step 3: Training the Generator and Encoder. Latent Vector Optimization')
        encoder = make_encoder()
        encoder.load_weights('encoder.h5')
        generator = make_generator()
        generator.load_weights('generator.h5')
        resize_model = make_resize()
        resize_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())

        face_rec = make_face_recognition()
        face_rec.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())
        face_rec.trainable = False

        image_input = Input(shape=(64, 64, 3))
        conditioning_variable = Input(shape=(6,))

        latent_approximation = encoder(image_input)
        reconstruction = generator([latent_approximation, conditioning_variable])

        # fr_resized_images = tf.image.resize(reconstruction, [192, 192], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        fr_resized_images = Lambda(lambda x: K.resize_images(x, height_factor=3, width_factor=3,
                                                          data_format='channels_last'))(reconstruction)
        embeddings = face_rec(fr_resized_images)

        fr_adversarial = Model(inputs=[image_input, conditioning_variable], outputs=[embeddings])
        fr_adversarial.compile(loss=euclidean_loss, optimizer=adv_opt)

        for epoch in range(epochs):
            print(f'Epoch: {epoch}')

            num_batches = int(len(loaded_images)/batch_size)
            for i in range(num_batches):
                batch = loaded_images[i*batch_size:(i+1)*batch_size]
                batch = batch / 127.5 - 1.
                batch = batch.astype(np.float32)
                y_batch = y[i*batch_size:(i+1)*batch_size]

                resized_image_batch = resize_model.predict_on_batch([batch])
                embeds = face_rec.predict_on_batch([resized_image_batch])
                loss = fr_adversarial.train_on_batch([batch, y_batch], embeds)

            print(f'Reconstruction Loss: {loss}')

            if epoch % 10 == 0:
                generator.save_weights('opt_generator.h5')
                encoder.save_weights('opt_encoder.h5')

        generator.save_weights('opt_generator.h5')
        encoder.save_weights('opt_encoder.h5')

if __name__ == '__main__':
    app.run(run_main)