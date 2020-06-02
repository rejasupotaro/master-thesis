import tensorflow as tf


def cross_entropy_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    output = tf.sigmoid(y_neg - y_pos)
    return tf.reduce_mean(output, axis=-1)


def hinge_loss(y_true, y_pred, margin=1.):
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    output = tf.maximum(0., margin + y_neg - y_pos)
    return tf.reduce_mean(output, axis=-1)
