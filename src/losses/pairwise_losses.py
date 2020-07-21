from typing import List

import numpy as np
import tensorflow as tf


def cross_entropy_loss(y_true: List[int], y_pred: List[float]):
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.reshape(y_true, [-1, 2])
    y_pred = tf.reshape(y_pred, [-1, 2])
    y_pred = tf.nn.softmax(y_pred) + np.finfo(float).eps
    loss = -(tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
    return loss


def hinge_loss(y_true: List[int], y_pred: List[float], margin: float = 1.):
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    loss = tf.maximum(0., margin - y_pos + y_neg)
    return tf.reduce_mean(loss, axis=-1)
