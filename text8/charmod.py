from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops.core_rnn import static_rnn

from ran import RANCell


class Model(object):

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.num_layers = num_layers = config.num_layers
    vocab_size = config.vocab_size
    self.in_size = in_size = config.hidden_sizes[0]
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    
    self.is_training = tf.placeholder(dtype=tf.bool, shape=[])
    keep_prob_x = 1 - (tf.to_float(self.is_training) * config.drop_x)
    keep_prob_o = 1 - (tf.to_float(self.is_training) * config.drop_o)

    embedding = tf.get_variable("embedding", [vocab_size, in_size])
    embedding = tf.nn.dropout(embedding, keep_prob_x, noise_shape=[vocab_size, 1])
    inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    def rancell(size):
      return tf.contrib.rnn.DropoutWrapper(RANCell(size), keep_prob_o)
    cell = tf.contrib.rnn.MultiRNNCell([rancell(s) for s in config.hidden_sizes[1:]])
    
    inputs = tf.unstack(inputs, num=num_steps, axis=1)
    self._initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, self._final_state = static_rnn(cell, inputs, self._initial_state)
    output = tf.reshape(tf.stack(outputs, axis=1), [-1, config.hidden_sizes[-1]])
    
    softmax_w = tf.transpose(embedding) if config.tied else tf.get_variable("softmax_w", [config.hidden_sizes[-1], vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
      [logits],
      [tf.reshape(self._targets, [-1])],
      [tf.ones([batch_size * num_steps])])
    pred_loss = tf.reduce_sum(loss) / batch_size
    self._cost = cost = pred_loss
    if not is_training:
      return
    tvars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars])
    self._cost = cost = pred_loss + config.weight_decay * l2_loss

    self._lr = tf.Variable(0.0, trainable=False)
    self._nvars = np.prod(tvars[0].get_shape().as_list())
    print(tvars[0].name, tvars[0].get_shape().as_list())
    for var in tvars[1:]:
      sh = var.get_shape().as_list()
      print(var.name, sh)
      self._nvars += np.prod(sh)
    print(self._nvars, 'total variables')
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def noise_x(self):
    return self._noise_x

  @property
  def noise_i(self):
    return self._noise_i

  @property
  def noise_h(self):
    return self._noise_h

  @property
  def noise_o(self):
    return self._noise_o

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def nvars(self):
    return self._nvars
