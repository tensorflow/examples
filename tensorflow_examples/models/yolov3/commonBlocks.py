import tensorflow as tf



class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated 
    """
    def call(self,x,training=False):
        if not training:
            training=tf.constant(False)
        training= tf.logical_and(training,self.trainable)
        return super().call(x,training)

def convolutional(input_layer,filters_shape,downsample=False,activate=True,bn=True):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1,0),(1,0))) (input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'
    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1],kernel=filters_shape[0] ,strides=strides,
                                 use_bais=not bn,kernel_regularizer=tf.keras.layers.regularizer.l2(0.0005),
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 bais_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv=BatchNormalization()(conv)
    if activate:
        conv=tf.nn.leaky_relu(conv, alpha=0.1)
    return conv
def residual_block(input_layer,input_channel,filter_num1,filter_num2):
    short_path=input_layer
    conv= convolutional(input_layer,filters_shape=(1,1,input_channel,filter_num1))
    conv= convolutional(conv,filters_shape=(3,3,filter_num1,filter_num2))

    return short_path+conv 
if __name__ == '__main__':
    #for debugging purposes
    pass