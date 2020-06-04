import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def cross_entropy_loss(y_true, y_pred, num_neg=1):
    logits = layers.Lambda(lambda a: a[::(num_neg + 1), :])(y_pred)
    labels = layers.Lambda(lambda a: a[::(num_neg + 1), :])(y_true)
    logits, labels = [logits], [labels]
    for neg_idx in range(num_neg):
        neg_logits = layers.Lambda(
            lambda a: a[neg_idx + 1::(num_neg + 1), :])(y_pred)
        neg_labels = layers.Lambda(
            lambda a: a[neg_idx + 1::(num_neg + 1), :])(y_true)
        logits.append(neg_logits)
        labels.append(neg_labels)
    logits = tf.concat(logits, axis=-1)
    labels = tf.concat(labels, axis=-1)
    smoothed_prob = tf.nn.softmax(logits) + np.finfo(float).eps
    loss = -(tf.reduce_sum(labels * tf.math.log(smoothed_prob), axis=-1))
    return loss


def hinge_loss(y_true, y_pred, margin=1.):
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    output = tf.maximum(0., margin + y_neg - y_pos)
    return tf.reduce_mean(output, axis=-1)
