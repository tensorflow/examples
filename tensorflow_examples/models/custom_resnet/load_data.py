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
"""Download the TinyImageNet dataset and get the ImageDataGenerator objects.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

class TinyImageNet(object):
	def __init__(self, img_size=64, train_size=10000, val_size=1000):
		self.img_size = img_size
		self.train_size = train_size
		self.val_size = val_size

	def download_data(self):
		"""Downloads the Stanford TinyImageNet dataset to the current directory.
		"""
		download_path = os.getcwd()
		path = tf.keras.utils.get_file('tiny-imagenet-200.zip', extract=True, cache_subdir=download_path,  
				origin='http://cs231n.stanford.edu/tiny-imagenet-200.zip')


	def train_val_gen(self, train_target=64, train_batch=64, val_target=64, val_batch=64):
		"""Instantiates and returns the Train and Val ImageDataGenerator objects.

		Args:
			train_target: Target size for the train geneartor. Tweak for progressive resizing.
			train_batch: Batch size for the Train generator.
			val_target: Target size for the val geneartor. Tweak for progressive resizing.
			val_batch: Batch size for the val generator.

		Returns:
			train_datagen: ImageDataGenerator object for training images. 
			val_datagen: ImageDataGenerator object for validation images.
		"""
		VAL_ANNOT = "tiny-imagenet-200/val/val_annotations.txt"
		TRAIN = "tiny-imagenet-200/train/"
		VAL = "tiny-imagenet-200/val/images"

		val_data = pd.read_csv(VAL_ANNOT , sep='\t', names=['File', 'Class', 'X', 'Y', 'H', 'W'])
		val_data.drop(['X','Y','H', 'W'], axis=1, inplace=True)

		train_datagen = ImageDataGenerator(
		        rescale=1./255,
		        rotation_range=18, # Rotation Angle
		        zoom_range=0.15,  # Zoom Range
		        width_shift_range=0.2, # Width Shift
		        height_shift_range=0.2, # Height Shift
		        shear_range=0.15,  # Shear Range
		        horizontal_flip=True, # Horizontal Flip
		        fill_mode="reflect", # Fills empty with reflections
		        brightness_range=[0.4, 1.6]  # Increasing/decreasing brightness
		)

		train_generator = train_datagen.flow_from_directory(
		        TRAIN,
		        target_size=(train_target, train_target),
		        batch_size=train_batch,
		        class_mode='categorical')

		val_datagen = ImageDataGenerator(rescale=1./255)

		val_generator = val_datagen.flow_from_dataframe(
		    val_data, directory=VAL, 
		    x_col='File', 
		    y_col='Class', 
		    target_size=(val_target, val_target),
		    color_mode='rgb', 
		    class_mode='categorical', 
		    batch_size=val_batch, 
		    shuffle=False, 
		    seed=42
		)

		return train_generator, val_generator

if __name__ == '__main__':
	ti = TinyImageNet()
	ti.download_data()