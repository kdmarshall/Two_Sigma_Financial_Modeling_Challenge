import tensorflow as tf
import numpy as np
import multiprocessing
import random
import os
from random import shuffle
import pandas as pd

from utils.data_utils import DataSet
from utils.mock_gym import r_score

import matplotlib.pyplot as plt # Required for saving out analytics
import matplotlib as mpl
mpl.use('TkAgg') # Backend for OSX -- change accordingly

max_seq_len = 50

num_features = 109#examples.shape[-1]
rnn_size = 512
rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
batch_size = 16

norm_init = tf.random_normal_initializer(0, 1)
embedding_weights = tf.get_variable('emb_w', [num_features, rnn_size*2], initializer=norm_init)
prediction_weights = tf.get_variable('pred_w', [rnn_size, 1], initializer=norm_init)
prediction_bias = tf.get_variable('pred_b', initializer=tf.constant(0.))

def relu(x, alpha=0., max_value=None):
    '''
    Note: when alpha != 0 this corresponds to leaky relu
    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x


observation_placeholder = tf.placeholder("float32", [batch_size, max_seq_len, 109])
targets_placeholder = tf.placeholder("float32", [batch_size, max_seq_len])
weights_placeholder = tf.placeholder("float32", [batch_size, max_seq_len])
#rewards_placeholder = tf.placeholder("float32", [batch_size, 1])

inputs = tf.transpose(observation_placeholder/2., [1, 0, 2])

outputs, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, time_major=True, scope='lstm', dtype=tf.float32)

logits = []
for timestep in tf.split(0, max_seq_len, outputs):
    pre_act = tf.matmul(tf.squeeze(timestep), prediction_weights) + prediction_bias
    #logit = relu(pre_act, 0.3)
    logit = tf.tanh(pre_act)
    logits.append(logit)

logits = tf.squeeze(tf.pack(logits))
logits = tf.transpose(logits, [1, 0])


loss = tf.reduce_sum(tf.square(tf.sub(logits, targets_placeholder*10.)) * weights_placeholder )#/ tf.reduce_sum(weights_placeholder))

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

def save_analytics(true, pred, r_values, step):

    plt.plot(range(len(true)), true)
    plt.plot(range(len(pred)), pred)
    plt.plot(range(len(pred)), r_values)
    plt.savefig('/Users/Peace/Desktop/outputs/example{0}.png'.format(step))
    plt.clf()


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    #avg = []

    #batch_input = np.random.normal(size=(batch_size, max_seq_len, 109))
    #batch_targets = np.random.normal(size=(batch_size, max_seq_len)) / 20.
    #batch_weights = np.ones((batch_size, max_seq_len))
    
    dataset = DataSet()
    
    data = dataset.get_numpy_data()
    
    trainset, train_validset, validset = dataset.split_valid(*data, 0.5)
    
    for step in range(1000000):
    
        input, targets, weights = dataset.get_numpy_batch(trainset,
                                                       batch_size, max_seq_len)

        l, _, logs = sess.run([loss, optimizer, logits],
                           feed_dict={
                            observation_placeholder: input,
                            targets_placeholder: targets,
                            weights_placeholder: weights})
        #avg.append(l)
        if step % 100 == 0:
            input, targets, weights = dataset.get_numpy_batch(validset,
                                                       batch_size, max_seq_len)
            l, logs = sess.run([loss, logits],
                               feed_dict={
                                observation_placeholder: input,
                                targets_placeholder: targets,
                                weights_placeholder: weights})
            #print('***')
            #print('Step {0}: {1}'.format(step, np.mean(avg)))
            print('Step {0}: {1}'.format(step, l))
            #print(logs)
            #avg = []
            r_values = []

            for v1, v2 in zip(np.rollaxis(targets, 1, 0), np.rollaxis(logs/10., 1, 0)):
                r_values.append(r_score(np.array(v1), np.array(v2)))

            save_analytics(targets[0]*10, logs[0], np.squeeze(np.array(r_values)), step)





            #saver.save(sess, "/Users/Peace/Projects/halite/models/production/model.ckp")




