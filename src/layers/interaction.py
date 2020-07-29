import itertools

import tensorflow as tf
from tensorflow.keras import layers


class WeightedInteraction(tf.keras.layers.Layer):
    def __init__(self, num_fields, **kwargs):
        self.num_fields = num_fields
        super(WeightedInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.constant_initializer(value=0)
        self.field_weights = tf.Variable(
            initial_value=w_init(shape=(self.num_fields, self.num_fields), dtype=tf.float32),
            constraint=tf.keras.constraints.NonNeg())
        super(WeightedInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        dim = inputs.shape[1] // self.num_fields
        interactions = []
        for i, j in itertools.combinations(range(self.num_fields), 2):
            interaction = layers.Dot(axes=1)([inputs[:, i * dim:(i + 1) * dim], inputs[:, j * dim:(j + 1) * dim]])
            interaction = tf.math.scalar_mul(self.field_weights[i, j], interaction)
            interactions.append(interaction)
        interactions = layers.Add()(interactions)
        return interactions

    def compute_output_shape(self, input_shape):
        return None, 1

    def get_config(self):
        config = super(WeightedInteraction, self).get_config().copy()
        config.update({
            'num_fields': self.num_fields,
        })
        return config
