import cPickle
import math
import time
import tensorflow as tf
import numpy as np
import os


def save_params(fname, saver, session):
    saver.save(session, fname)


def load_data(fName):
    f = file(fName,'rb')
    obj = cPickle.load(f)
    f.close()
    return obj


def compute_mean_abs_norm(grads_and_vars):
    tot_grad = 0
    tot_w = 0
    N = len(grads_and_vars)

    for g, w in grads_and_vars:
        tot_grad += tf.reduce_mean(tf.abs(g))
        tot_w += tf.reduce_mean(tf.abs(w))
    return tot_grad/N, tot_w/N


def relu(x, alpha=1./5.5):
    return tf.maximum(alpha * x, x)


def normalize(x, mean, std):
    return (x - mean)/std


def denormalize(x, mean, std):
    return x * std + mean


