# Mask R-CNN
### Migration Guide
#
#### Making the code Tensorflow 2.0 native
---
#
#### 1. Rehoming Keras layers to tf.keras.layers and removing tf.contrib calls
To migrate the model to tf.keras.layers, all instances of keras.layers is replaced with tf.keras.layers. The Input layer keras.layers is rehomed to tf.keras.Input. Specify the input_shape to be able obtain model summaries. Here, there were no tf.contrib calls, so the migration involves few changes due to rehoming and renaming of functions.
The following changes ensued:
- Replace `keras.layers` calls with `tf.keras.layers`
- Replace `keras.models.Model` with `tf.keras.Model` which is the updated class definition  
#  
#### 2. Remove depracated APIs and update their function calls
The topology class from keras.engine has been depracated and some functionality has been merged with the saving class. The `saving.load_weights_from_hdf5_group(f, layers) ` is an efficient implementation of loading specific layers, unfortunately was not reiplmented with later keras versions and hence needs a workaround [[Idea](https://github.com/keras-team/keras/issues/1873)].  
The following changes ensued:
- Remove all calls to keras.engine.topology and keras.engine.saving
- Replace `keras.engine.layer` with `tf.keras.layers.Layer`
- Replace `saving.load_weights_from_hdf5_group(filepath, layers)` with `tf.keras.Model.load_weights(filepath, by_name=False)`  
#  
#### 3. Math operations are rehomed to tf.math 
All the math operations have been rehomed to `tf.math` and need to be replaced from previous `tf.*` calls. This model had many math especially in the ROI pooling and FPN(Feature Pyramid Network) calculations. The following calls were rehomed:
- Min-Max operations: `tf.math.maximum`, `tf.math.minimum`, `tf.math.argmax`, `tf.math.argmin`
- Compare operations: `tf.math.greater`, `tf.math.logical_and`, `tf.math.equal`
- Dimension operations: `tf.math.reduce_mean`, `tf.math.reduce_sum`
#
#### 4. Choose between using the Functional API and Subclassing the model
The Functional API is the best way to create models with `tf.keras`. Subclassing the model provides more flexibility to experiment with and allows using loops and conditionals to create different models. This is good for R&D but the model cannot be exported.  
For the subclassing, there are 3 main functions:
- Collect layer parameters in __init__ and define all necessary variables.
- Build the variables in build. It is called once and is useful in setting the shape or size of the input.
- Execute the calculations in call, and return the result (Using the Functional API).  
Here, the Encoder and Decoder are subclasses of `tf.keras.layers.Layer` and are used in the Autoencoder model which is a subclass of `tf.keras.Model`.
#   
#### 5. Remove tf.placeholders and replace tf.Session.run calls
Every `tf.Session.run` call should be replaced by a Python function.
- The `feed_dict` and `tf.placeholders` become function arguments.
- The fetches become the function's return value.
- The regularizations are calculated manually, without referring to any global collection.
You can step-through and debug the function using standard Python tools like pdb. 
#
#### 6. Using the tf.function decorator
Although tf 2.0 is eager by default, graph mode execution is the most efficient way to execute a model. The `tf.function` decorator marks the function for JIT compilation using Tensorflow graphs. This uses [AutoGraph](https://render.githubusercontent.com/view/autograph.ipynb) which also ensures execution in contexts (for tflite, tfjs...etc) and manages control dependencies.   
Here, the decorator was used in the `call()` function of Autoencoder.   
#

