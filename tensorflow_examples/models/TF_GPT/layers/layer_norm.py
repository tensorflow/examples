import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.gamma = self.add_weight(
            "layer_norm_scale",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.ones_initializer(),
            experimental_autocast=False)
        self.beta = self.add_weight(
            "layer_norm_bias",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.zeros_initializer(),
            experimental_autocast=False)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, epsilon=1e-6, input_dtype=tf.float32):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        normalized = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return tf.cast(normalized * self.gamma + self.beta, input_dtype)
