# Autoencoder - Migration Guide
### Making the code Tensorflow 2.0 native
##### 1. Remove tf.contrib calls   
The contrib module is no longer available in TensorFlow 2.0 even with tf.compat.v1. Some tf.contrib layers might not have been moved to core TensorFlow but have instead been moved to the [TF add-ons ](https://github.com/tensorflow/addons). The best way to migrate would be to convert tf.contrib.layers to tf.layers, being mindful of the different function signatures. Migrate the tf.layers to keras.
Here, there were no tf.contrib calls, so a few changes due to rehoming and renaming of functions.
#   
##### 2. Choose between using the Functional API and Subclassing the model
The Functional API is the best way to create models with `tf.keras`. Subclassing the model provides more flexibility to experiment with and allows using loops and conditionals to create different models. This is good for R&D but the model cannot be exported.  
For the subclassing, there are 3 main functions:
- Collect layer parameters in __init__ and define all necessary variables.
- Build the variables in build. It is called once and is useful in setting the shape or size of the input.
- Execute the calculations in call, and return the result (Using the Functional API).  
Here, the Encoder and Decoder are subclasses of `tf.keras.layers.Layer` and are used in the Autoencoder model which is a subclass of `tf.keras.Model`.
#   
##### 3. Remove tf.placeholders and replace tf.Session.run calls
Every `tf.Session.run` call should be replaced by a Python function.
- The `feed_dict` and `tf.placeholders` become function arguments.
- The fetches become the function's return value.
- The regularizations are calculated manually, without referring to any global collection.
You can step-through and debug the function using standard Python tools like pdb.   
#
##### 4. Using the tf.function decorator
Although tf 2.0 is eager by default, graph mode execution is the most efficient way to execute a model. The `tf.function` decorator marks the function for JIT compilation using Tensorflow graphs. This uses [AutoGraph](https://render.githubusercontent.com/view/autograph.ipynb) which also ensures execution in contexts (for tflite, tfjs...etc) and manages control dependencies.   
Here, the decorator was used in the `call()` function of Autoencoder.   
#
##### 5. tf.data input pipelines
The recommended way to feed data to a model is to use the tf.data package, which contains a collection of high performance classes for manipulating data.   
Here, `tf.data.Dataset.from_tensor_slices` was used to batch and shuffle the data.
#
##### 6. Training
There are two ways to train the model. Eager mode training with `tf.GradientTape` and Graph mode training with `model.fit`. The eager mode training is better suited for fast prototyping and uses the autodifferentiation with `tape.gradient()` calls. The graph mode training is much faster with the computation graph and can make use of distributed training as well. They support callbacks like tf.keras.callbacks.TensorBoard, and custom callbacks and are hence recommended.    
Here, the training is done using tf.GradientTape with the Adam optimizer and mean squared error loss to facilitate quick prototyping.
