import tensorflow as tf


class AddBias0(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddBias0, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=[1])
        super(AddBias0, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.math.add(inputs, self.bias)
