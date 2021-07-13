import math
import numpy as np
import tensorflow as tf
import os
import time


class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            pi = tf.contrib.layers.fully_connected(inputs, self.a_dim)
        return pi
    def __init__(self, state_dim, action_dim, learning_rate):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.sess = tf.Session()
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])   
        self.pi = self.CreateNetwork(inputs=self.inputs)
        self.sess.run(tf.global_variables_initializer())
    