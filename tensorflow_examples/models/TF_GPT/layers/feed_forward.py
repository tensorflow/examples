from utils.tf_utils import *


class Conv1d(tf.keras.layers.Layer):
    def __init__(self, hidden_size,
                 filter_size,
                 weights_init_stdev=0.02,
                 weights_mean=0.0,
                 bias_init=0.0):
        super(Conv1d, self).__init__()

        self.weights_init_stdev = weights_init_stdev
        self.weights_mean = weights_mean
        self.bias_init = bias_init
        self.hidden_size = hidden_size
        self.filter_size = filter_size

    def build(self, input_shape):
        self.weight = self.add_weight(
            "cov1d_weights",
            shape=[self.hidden_size, self.filter_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                stddev=self.weights_init_stdev,
                mean=self.weights_mean))

        self.bias = self.add_weight("conv1d_biases",
                                    shape=[self.filter_size],
                                    initializer=tf.constant_initializer(self.bias_init))
        super(Conv1d, self).build(input_shape)

    def call(self, inputs):
        output_shape = [tf.shape(inputs)[0], tf.shape(inputs)[1]] + [self.filter_size]
        inputs = tf.reshape(inputs, [-1, self.hidden_size])  # shape [batch, seq , features] => [batch*seq, features]
        outputs = tf.matmul(inputs, self.weight) + self.bias
        outputs = tf.reshape(outputs, output_shape)  # Reshape => [batch, seq, filter_size]
        return outputs


class FeedForward(tf.keras.layers.Layer):

    def __init__(self, hidden_size, filter_size, dropout_rate=0.1, activation=tf.nn.relu):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.dense_layer = Conv1d(self.hidden_size, self.filter_size)
        self.output_dense_layer = Conv1d(self.filter_size, self.hidden_size)

    def call(self, x, training=False):
        output = self.dense_layer(x)
        output = self.activation(output)
        output = self.output_dense_layer(output)

        if training:
            output = tf.nn.dropout(output, rate=self.dropout_rate, name="feed_forward_dropout")

        return output
