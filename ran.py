from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class RANCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, use_tanh=True, num_proj=None):
        self.num_units = num_units
        self.use_tanh = use_tanh
        self.num_proj = num_proj


    @property
    def output_size(self):
        if self.num_proj is not None:
            return self.num_proj
        else:
            return self.num_units


    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.num_units, self.output_size)


    def linear(self, inputs, output_size, use_bias=True):
        w = tf.get_variable("w", [inputs.get_shape()[-1].value, output_size])
        if use_bias:
            b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer())
            return tf.nn.xw_plus_b(inputs, w, b)
        else:
            return tf.matmul(inputs, w)


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state
            with tf.variable_scope("content"):
                content = self.linear(inputs, self.num_units, use_bias=False)
            with tf.variable_scope("gates"):
                gates = tf.nn.sigmoid(self.linear(tf.concat([inputs, h], 1), 2 * self.num_units))

            i, f = tf.split(gates, num_or_size_splits=2, axis=1)
            new_c = i * content + f * c
            new_h = new_c
            if self.use_tanh:
                new_h = tf.tanh(new_h)
            if self.num_proj is not None:
                new_h = self.linear(new_h, self.num_proj)
            output = new_h
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            return output, new_state
