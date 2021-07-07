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
"""Runner Script.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from resnet import ResNet
from load_data import TinyImageNet
import tensorflow as tf
assert tf.__version__.startswith('2.')

from tensorflow.keras.callbacks import Callback, LearningRateScheduler

class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=5, startAt=0):
		super(EpochCheckpoint, self).__init__()
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt

	def on_epoch_end(self, epoch, log={}):
		if (self.intEpoch+1) % self.every == 0:
			path = os.path.sep.join([self.outputPath, 
					"custom_resnet.hdf5".format(self.intEpoch+1)])
			self.model.save(path, overwrite=True)
		self.intEpoch += 1


def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    
    return alpha

def run_main(argv):
	del argv
	kwargs = {}
	main(**kwargs)

def main():
	ti = TinyImageNet()
	model = ResNet.build(None, None, 3, 200, (3, 4, 6), (64, 128, 256, 512), reg=0.0005)
	print(f'Custom ResNet model built.')

	callbacks = [EpochCheckpoint("./checkpoints/", every=5),
	        LearningRateScheduler(poly_decay)]

	opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.1, amsgrad=False)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	print(f'Model compiled. Training on 64x64 sized images upcoming.')

	train_gen, val_gen = ti.train_val_gen(train_target=64, train_batch=64, val_target=64, val_batch=64)

	model.fit_generator(
		  train_gen,
		  steps_per_epoch=100000 // 64,
		  validation_data=val_gen,
		  validation_steps=10000 // 64,
		  epochs=20,
		  max_queue_size=64 * 2,
		  callbacks=callbacks,
		  verbose=1
		)

	# Save the model
	filepath = "./checkpoints/epoch_20_64.hdf5"

	model.save(
	    filepath,
	    overwrite=True,
	    include_optimizer=True
	)
	print(f'Training for 20 epochs on 64x64 sized images has completed. Total Epochs: 20')

	# Continue training with 32x32 sized images.
	train_gen, val_gen = ti.train_val_gen(train_target=32, train_batch=64, val_target=64, val_batch=64)

	model.fit_generator(
	  train_gen,
	  steps_per_epoch=100000 // 64,
	  validation_data=val_gen,
	  validation_steps=10000 // 64,
	  epochs=20,
	  max_queue_size=128,
	  callbacks=callbacks,
	  verbose=1
	)

	filepath = "./checkpoints/epoch_40_32.hdf5"

	model.save(
	    filepath,
	    overwrite=True,
	    include_optimizer=True
	)
	print(f'Training for 20 epochs on 32x32 sized images has completed. Total Epochs: 40')

	# Continue training with 16x16 sized images.
	train_gen, val_gen = ti.train_val_gen(train_target=16, train_batch=64, val_target=64, val_batch=64)

	model.fit_generator(
	  train_gen,
	  steps_per_epoch=100000 // 64,
	  validation_data=val_gen,
	  validation_steps=10000 // 64,
	  epochs=20,
	  max_queue_size=64,
	  callbacks=callbacks,
	  verbose=1
	)

	# Save the model
	filepath = "./checkpoints/epoch_60_16.hdf5"

	model.save(
	    filepath,
	    overwrite=True,
	    include_optimizer=True
	)
	print(f'Training for 20 epochs on 16x16 sized images has completed. Total Epochs: 60')

	# Continue training with 32x32 sized images.
	train_gen, val_gen = ti.train_val_gen(train_target=32, train_batch=64, val_target=64, val_batch=64)

	model.fit_generator(
	  train_gen,
	  steps_per_epoch=100000 // 64,
	  validation_data=val_gen,
	  validation_steps=10000 // 64,
	  epochs=20,
	  max_queue_size=64,
	  verbose=1
	)

	# Save the model
	filepath = "./checkpoints/epoch_80_32.hdf5"

	model.save(
	    filepath,
	    overwrite=True,
	    include_optimizer=True
	)
	print(f'Training for another 20 epochs on 32x32 sized images has completed. Total Epochs: 80')

	# Continue training with 64x64 sized images.
	train_gen, val_gen = ti.train_val_gen(train_target=64, train_batch=64, val_target=64, val_batch=64)

	model.fit_generator(
	  train_gen,
	  steps_per_epoch=100000 // 64,
	  validation_data=val_gen,
	  validation_steps=10000 // 64,
	  epochs=20,
	  max_queue_size=64,
	  verbose=1
	)

	# Save the model
	filepath = "./checkpoints/epoch_100_64.hdf5"

	model.save(
	    filepath,
	    overwrite=True,
	    include_optimizer=True
	)
	print(f'Training for another 20 epochs on 64x64 sized images has completed. Total Epochs: 100')

if __name__ == '__main__':

	NUM_EPOCHS = 30
	INIT_LR = 0.01
	app.run(run_main)
	