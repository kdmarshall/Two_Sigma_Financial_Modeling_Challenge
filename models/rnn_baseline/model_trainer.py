import tensorflow as tf
import numpy as np
import multiprocessing
import random
import os
from random import shuffle
import pandas as pd

from utils.data_utils import DataSet
from utils.mock_gym import r_score

# Training options
SAVE_ANALYTICS = False
OUTDIR = '/Users/Peace/Desktop/outputs'

if SAVE_ANALYTICS:
    import matplotlib.pyplot as plt # Required for saving out analytics
    import matplotlib as mpl
    mpl.use('TkAgg') # Backend for OSX -- change accordingly

def save_analytics(true, pred, r_values, step):
    plt.plot(range(len(true)), true)
    plt.plot(range(len(pred)), pred)
    plt.plot(range(len(pred)), r_values)
    plt.savefig(OUTDIR + '/example{0}.png'.format(step))
    plt.clf()

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

print('initializing...')

# Hyperparameters
max_seq_len = 30
num_features = 109 # TODO: examples.shape[-1]
rnn_size = 8
batch_size = 128
learning_rate = 1e-3
num_steps = 100000
valid_steps = 100

# Initialize TF variables
rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
embedding_weights = tf.get_variable('emb_w', [num_features, rnn_size], initializer=tf.contrib.layers.xavier_initializer())
p_l1_weights = tf.get_variable('pred_l1_w', [rnn_size, 4], initializer=tf.contrib.layers.xavier_initializer())
p_l1_bias = tf.get_variable('pred_l1_b', initializer=tf.constant(0.))
prediction_weights = tf.get_variable('pred_w', [4, 1], initializer=tf.contrib.layers.xavier_initializer())
prediction_bias = tf.get_variable('pred_b', initializer=tf.constant(0.))

# Input nodes into the graph
observation_placeholder = tf.placeholder("float32", [batch_size, max_seq_len, 109])
targets_placeholder = tf.placeholder("float32", [batch_size, max_seq_len])
weights_placeholder = tf.placeholder("float32", [batch_size, max_seq_len])
#rewards_placeholder = tf.placeholder("float32", [batch_size, 1])

def get_graph():
    inputs = tf.transpose(observation_placeholder, [1, 0, 2])

    embedded = []
    for input in tf.unpack(inputs, axis=0):
        act = tf.matmul(input, embedding_weights)
        embedded.append(act)

    outputs, _ = tf.nn.dynamic_rnn(rnn_cell, tf.pack(embedded), time_major=True, scope='lstm', dtype=tf.float32)

    logits = []
    for timestep in tf.split(0, max_seq_len, outputs):
        pre_act_l1 = tf.matmul(tf.squeeze(timestep), p_l1_weights) + p_l1_bias
        act_l1 = relu(pre_act_l1, 0.3)
        pre_act_l2 = tf.matmul(act_l1, prediction_weights) + prediction_bias
        logit = tf.tanh(pre_act_l2)
        logits.append(logit)

    logits = tf.squeeze(tf.pack(logits))
    logits = tf.transpose(logits, [1, 0])

    # R is differentiable, so we can optimize the evaluation function directly
    y_true = targets_placeholder*10. # Scale to take adv of full tanh range
    diffs = tf.square(y_true - logits) * weights_placeholder
    y_true_mean = tf.reduce_sum(y_true)/tf.reduce_sum(weights_placeholder)
    denom = tf.reduce_sum(tf.square(y_true - y_true_mean) * weights_placeholder)
    R2 = 1 - tf.reduce_sum(diffs) / (denom + 1e-17)
    loss = -1 * tf.sign(R2) * tf.sqrt(tf.abs(R2)) # -1 to maximize R

    # SSE loss
    #loss = tf.reduce_sum(tf.square(tf.sub(logits, targets_placeholder*10.)) * weights_placeholder )#/ tf.reduce_sum(weights_placeholder))

    return logits, loss

logits, loss = get_graph()

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # Useful for testing overfit model
    #batch_input = np.random.normal(size=(batch_size, max_seq_len, 109))
    #batch_targets = np.random.normal(size=(batch_size, max_seq_len)) / 20.
    #batch_weights = np.ones((batch_size, max_seq_len))
    
    dataset = DataSet()
    
    data = dataset.get_numpy_data()

    trainset, train_validset, validset = dataset.split_valid(*data, 0.25)
    
    print('Train dataset shape: {0}'.format(trainset[0].shape))
    # trainset stats:
    # shape: 712, 1520 when split 0.5
    # Epoch about every ~8000 steps (not true epoch due to shifted seq)
    
    print('training...')
    print('Format: Train R -- Valid R')
    avg = []
    for step in range(num_steps):
    
        input, targets, weights = dataset.get_numpy_batch(trainset,
                                                       batch_size, max_seq_len)
        # Allow for burn-in
        weights[:5] = 0

        l, _, logs = sess.run([loss, optimizer, logits],
                           feed_dict={
                            observation_placeholder: input,
                            targets_placeholder: targets,
                            weights_placeholder: weights})
        avg.append(-l)
        if step % 1000 == 0:
            vavg = []
            for vstep in range(valid_steps):
                input, targets, weights = dataset.get_numpy_batch(validset,
                                                                   batch_size,
                                                                   max_seq_len)
                weights[:5] = 0

                l, logs = sess.run([loss, logits],
                                   feed_dict={
                                    observation_placeholder: input,
                                    targets_placeholder: targets,
                                    weights_placeholder: weights})
            
                vavg.append(-l)

            print('Step {0}: {1:.4f} {2:.4f}'.format(step, np.mean(avg), np.mean(vavg)))

            avg = []

            if SAVE_ANALYTICS:
                r_values = []

                for v1, v2 in zip(np.rollaxis(targets, 1, 0), np.rollaxis(logs/10., 1, 0)):
                    r_values.append(r_score(np.array(v1), np.array(v2)))
                save_analytics(targets[0]*10, logs[0], np.squeeze(np.array(r_values)), step)

            #saver.save(sess, OUTDIR+'/models/model.ckp")



