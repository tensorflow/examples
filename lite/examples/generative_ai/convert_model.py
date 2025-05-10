import tensorflow as tf
from keras_nlp.models import GPT2CausalLM

# Load a pre-trained model (e.g., GPT-2 for text generation)
model = GPT2CausalLM.from_preset("gpt2_base_en")
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
# Save as autocomplete.tflite
with open("autocomplete.tflite", "wb") as f:
    f.write(tflite_model)