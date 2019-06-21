import numpy as np
import tensorflow as tf


def softmax(weight):
    exp = np.exp(weight)
    return exp / np.sum(exp)


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def masked_softmax(x, mask):
    x = x - tf.reduce_max(x, keep_dims=True)
    x = tf.exp(x)
    if mask is not None:
        x = x * mask
    x = x / (tf.reduce_sum(x, keep_dims=True))
    return x

