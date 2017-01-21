import tensorflow as tf
import numpy as np
import multiprocessing
import random
import os
from random import shuffle
import pandas as pd
import h5py
from scipy.integrate import simps
import warnings
from sklearn.metrics import r2_score

DEBUG = True
RUN = False

if DEBUG:
    PROJECT_DIR = os.path.dirname(
                    os.path.dirname(
                        os.path.realpath(__file__)))
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'train.h5').replace('models/', '')

    from utils import mock_gym as kagglegym

else:
    TRAIN_DATA_FILE = '../input/train.h5'
    import kagglegym



RANDOM_SEED = 8888

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

def r_score(y_true, y_pred, sample_weight=None, multioutput=None):

    # SKL is not self-consistent. Filter out the many deprecation warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        r2 = r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
        r = (np.sign(r2)*np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r



class DataSet(object):
    """class for dataset processing"""
    def __init__(self, path=TRAIN_DATA_FILE):
        self.path = path
        self.data_dict = self._get_data_dict()
        self.df = self._get_df()
    
        self.col_means = None
        self.col_stds = None
        self.cols = []

    def _get_data_dict(self):
        with h5py.File(self.path,'r') as hf:
            train_hf = hf.get('train')
            data_dict = { hf_key: np.array(train_hf.get(hf_key))
                            for hf_key in train_hf.keys()}
            return data_dict

    def _get_df(self):
        with pd.HDFStore(self.path, "r") as train:
            df = train.get("train")
            return df

    def __repr__(self):
        sets = [ "{}: {}".format(key,data_set.shape) 
                    for key, data_set in 
                        self.data_dict.iteritems()]
        return "; ".join(sets)

    def keys(self):
        return self.data_dict.keys()

    def get(self, key):
        return self.data_dict.get(key, None)

    def to_df(self):
        return self.df

    def get_batch(self, slice_index, batch_size, columns=None, random=False):
        if random:
            samples = self.df.sample(n=batch_size)
        else:
            num_samples = self.df.shape[0]
            if (slice_index+1)*batch_size >= num_samples:
                print("Slice is out of range. Taking last batch_size slice")
                sample_range = (num_samples - batch_size, num_samples)
            else:
                sample_range = (slice_index*batch_size, (slice_index+1)*batch_size)
            samples = self.df[sample_range[0] : sample_range[1]]
        samples_matrix = np.array(samples.as_matrix(columns=columns)) if columns else np.array(samples.as_matrix())
        return samples_matrix

    def get_numpy_data(self):

        df = self.df
        
        # Let's limit the data for now
        features = ['technical_20', 'technical_30']
        meta = ['y', 'timestamp', 'id']
        df = df[features+meta]
        
        means = []
        stds = []
        
        # Assuming column order remains consistent throughout the class
        for col in df.columns:
            if col not in ['y', 'timestamp', 'index', 'id']:
                data = df[col].dropna().as_matrix()
                means.append(np.mean(data))
                stds.append(np.std(data))
                self.cols.append(col)
    
        self.col_means = np.array(means)
        self.col_stds = np.array(stds)

        # Ensure values are sorted by time
        df = df.sort_values(by=['id', 'timestamp'], ascending=True)

        max_seq_len_raw = 1820

        # Simply mean-fill missing values for now
        df = df.fillna(df.mean())
        
        ids = np.unique(df['id'].as_matrix())
        examples = []
        targets = []
        weights = []
        for id in ids:
            slice = df[df.id == id]
            num_timesteps = slice.shape[0]

            #y = slice['y'].as_matrix()
            
            # Pad df to max seq len
            padded = slice.reset_index().reindex(range(max_seq_len_raw),
                                                                fill_value=0)
            target = padded['y'].as_matrix()
            padded.drop('y', axis=1, inplace=True)
            padded.drop('timestamp', axis=1, inplace=True)
            padded.drop('index', axis=1, inplace=True)
            padded.drop('id', axis=1, inplace=True)
            
            example = padded.as_matrix()
            
            examples.append(example)
            targets.append(target)
            weight = [1]*num_timesteps + [0]*(max_seq_len_raw - num_timesteps)

            weights.append(weight)

        examples = np.array(examples)
        targets = np.array(targets)
        weights = np.array(weights)
        
        # Normalize the data
        #examples = (examples - self.col_means)/self.col_stds
        
        # TODO: Supply these outside the function later: col_means, col_stds

        return examples, targets, weights
        
    def normalize(self, data):
        return (data - self.col_means)/self.col_stds

    def split_valid(self, examples, targets, weights, valid_split_ratio=0.5):
        """
        Args:
            valid_split_ratio: float range 0-1.; percentage of data reserved
            for validation. Note that two validation sets are reserved: unique
            ids are reserved entirely for validation, and, latter timesteps for
            sequences used in training are also used in validation.
        
        """

        num_ids = examples.shape[0]

        valid_num = int(round(num_ids*valid_split_ratio))

        examples_train_pre = examples[:-valid_num]
        targets_train_pre = targets[:-valid_num]
        weights_train_pre = weights[:-valid_num]

        examples_valid = examples[-valid_num:]
        targets_valid = targets[-valid_num:]
        weights_valid = weights[-valid_num:]

        examples_train = []
        targets_train = []
        weights_train = []
        
        examples_train_valid = []
        targets_train_valid = []
        weights_train_valid = []
        
        valid_len = 900 # Hardcoded for now

        for arr1, arr2, arr3 in zip(examples_train_pre, targets_train_pre,
                                                            weights_train_pre):

            examples_train.append(arr1[:-valid_len])
            targets_train.append(arr2[:-valid_len])
            weights_train.append(arr3[:-valid_len])
            
            examples_train_valid.append(arr1[-valid_len:])
            targets_train_valid.append(arr2[-valid_len:])
            weights_train_valid.append(arr3[-valid_len:])

        trainset = (np.array(examples_train), np.array(targets_train),
                                                    np.array(weights_train))
        train_validset = (np.array(examples_train_valid),
                                                np.array(targets_train_valid),
                                                np.array(weights_train_valid))

        validset = (examples_valid, targets_valid, weights_valid)

        return trainset, train_validset, validset

    def get_numpy_batch(self, dataset, batch_size, seq_len):
        examples = []
        targets = []
        weights = []
        
        #for _ in range(batch_size):
        while len(targets) < batch_size:
            # Sample a random id
            idx = np.random.choice(range(dataset[0].shape[0]))
            
            # Take random slice
            max_seq_len = dataset[0][idx].shape[0]
            
            assert max_seq_len >= seq_len
            slice = np.random.choice(range(max_seq_len - seq_len))
            
            # Let's just go with full length for now
            w = dataset[2][idx][slice:slice+seq_len]
            if np.sum(w) != len(w):
                continue
            
            examples.append(dataset[0][idx][slice:slice+seq_len])
            targets.append(dataset[1][idx][slice:slice+seq_len])
            weights.append(w)

        return np.array(examples), np.array(targets), np.array(weights)


    def preprocess_timestep(self, data):
        ids = data['id'].as_matrix()
        data = data.copy()
        data.drop('timestamp', axis=1, inplace=True)
        data.drop('id', axis=1, inplace=True)
        
        for ix, col in enumerate(self.cols):
            data[col] = data[col].fillna(self.col_means[ix])

        data = data.as_matrix()
        data = (data - self.col_means)/self.col_stds

        return data, ids

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
num_features = 2#108 # TODO: examples.shape[-1]
rnn_size = 512
p_l1_size = 128
batch_size = 128*10
learning_rate = 1e-4
num_steps = 100000
valid_steps = 300
split_ratio = 0.5 # % of ids reserved for validation
keep_prob = 1 # Only used during training

# Initialize TF variables
rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
embedding_weights = tf.get_variable('emb_w', [num_features, rnn_size], initializer=tf.contrib.layers.xavier_initializer())
p_l1_weights = tf.get_variable('pred_l1_w', [rnn_size, p_l1_size], initializer=tf.contrib.layers.xavier_initializer())
p_l1_bias = tf.get_variable('pred_l1_b', initializer=tf.constant(0., shape=[p_l1_size]))
prediction_weights = tf.get_variable('pred_w', [p_l1_size, 1], initializer=tf.contrib.layers.xavier_initializer())
prediction_bias = tf.get_variable('pred_b', initializer=tf.constant(0.))

# Input nodes into the graph
observation_placeholder = tf.placeholder("float32", [None, max_seq_len, num_features])
targets_placeholder = tf.placeholder("float32", [None, max_seq_len])
weights_placeholder = tf.placeholder("float32", [None, max_seq_len])
#rewards_placeholder = tf.placeholder("float32", [batch_size, 1])
keep_prob_placeholder = tf.placeholder(tf.float32)

def get_graph():
    inputs = tf.transpose(observation_placeholder, [1, 0, 2])

    embedded = []
    for input in tf.unpack(inputs, axis=0):
        act = tf.nn.dropout(tf.matmul(input, embedding_weights), keep_prob_placeholder)
        embedded.append(act)

    outputs, _ = tf.nn.dynamic_rnn(rnn_cell, tf.pack(embedded), time_major=True, scope='lstm', dtype=tf.float32)

    logits = []
    for timestep in tf.split(0, max_seq_len, outputs):
        pre_act_l1 = tf.matmul(tf.squeeze(timestep), p_l1_weights) + p_l1_bias
        act_l1 = tf.nn.dropout(relu(pre_act_l1, 0.3), keep_prob_placeholder)
        pre_act_l2 = tf.matmul(act_l1, prediction_weights) + prediction_bias
        logit = tf.tanh(pre_act_l2)
        logits.append(logit)

    logits = tf.squeeze(tf.pack(logits))
    logits = tf.transpose(logits, [1, 0])

    # R is differentiable, so we can optimize the evaluation function directly
    y_true = targets_placeholder
    diffs = tf.square(y_true - logits/10.) * weights_placeholder # Scale to take adv of full tanh range
    y_true_mean = tf.reduce_sum(y_true * weights_placeholder)/tf.reduce_sum(weights_placeholder)
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
    
    examples, targets, weights = dataset.get_numpy_data()

    
    del dataset.df
    
    examples = dataset.normalize(examples)

    trainset, train_validset, validset = dataset.split_valid(examples, targets, weights, split_ratio)
    
    del examples
    del targets
    del weights
    
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
        weights[:-2] = 0

        l, _, logs = sess.run([loss, optimizer, logits],
                           feed_dict={
                            observation_placeholder: input,
                            targets_placeholder: targets,
                            weights_placeholder: weights,
                            keep_prob_placeholder: keep_prob})
        avg.append(-l)
        #if DEBUG or RUN: # Don't need to validate during submission
        if step % 200 == 0 and step > -1:
            vavg = []
            y_trues = []
            y_hats = []
            for vstep in range(int(round(12000/batch_size))):
                input, targets, weights = dataset.get_numpy_batch(validset,
                                                                   batch_size,
                                                                   max_seq_len)
                weights[:-2] = 0

                l, logs = sess.run([loss, logits],
                                   feed_dict={
                                    observation_placeholder: input,
                                    targets_placeholder: targets,
                                    weights_placeholder: weights,
                                    keep_prob_placeholder: 1.0})
            
                vavg.append(-l)
                y_hats += list(logs[:, -1]/10.)
                y_trues += list(targets[:, -1])
                
            scores = []
            areas = []
            for i in range(20, len(y_hats)):
                scores.append(r_score(y_trues[:i], y_hats[:i]))
                area = simps(scores, dx=1)
                areas.append(area)
            if False:#DEBUG:
                np.save('/Users/Peace/Desktop/truesT', np.array(y_trues))
                np.save('/Users/Peace/Desktop/hatsT', np.array(y_hats))
                np.save('/Users/Peace/Desktop/areasT', np.array(areas))
                saver.save(sess, '/Users/Peace/Desktop/temp3.ckp')
            
            # Exponential decay to help with metric stability problem
            scores_ed = []
            for i in range(len(scores)):
                scores_ed.append(scores[i]*(0.98**i))
            area_ed = simps(scores_ed, dx=1)

            print('Step {0}: {1:.4f} {2:.4f} {3:.4f} {4:.4f}'.format(step, np.mean(avg), np.mean(vavg), scores[-1], area_ed)) # Area is current. We want to know the final area here only.

            avg = []
            
            if np.mean(vavg) > 0 and scores[-1] > 0 and area_ed > 0:
                break


            # Rudimentary early stopping for now (TODO: Learning rate decay;
            # conditional model saving)
            #if np.mean(vavg) > 0.018 or step == 1800:
            #    break

            # For debugging
            #if DEBUG:
            #    saver.restore(sess, '/Users/Peace/Desktop/temp2.ckp')
            #    break


                    




    if False:
        y_trues = []
        y_hats = []
        # Run a bunch of validation steps to assess volatility of R
        for vstep in range(int(round(2000/batch_size))):
            input, targets, weights = dataset.get_numpy_batch(validset,
                                                               batch_size,
                                                               max_seq_len)


            logs = sess.run([logits],
                               feed_dict={
                                observation_placeholder: input,
                                keep_prob_placeholder: 1.0})[0]
            y_hats += list(logs[:, -1]/10.)
            y_trues += list(targets[:, -1])

        #print('trues:')
        #for item in y_trues:
        #    print(item)
        #print('hats:')
        #for item in y_hats:
        #    print(item)
        #import matplotlib.pyplot as plt
        np.save('/Users/Peace/Desktop/trues', np.array(y_trues))
        np.save('/Users/Peace/Desktop/hats', np.array(y_hats))
    #mbjh
    del trainset
    del train_validset
    del validset

    env = kagglegym.make()
    obs = env.reset()

    # Now that training is complete, we can start predicting the target
    history = {}
    running_seq = []
    rewards = []
    #print('Average reward over time:')
    while True:
        data, ids = dataset.preprocess_timestep(obs.features)
        # Unfortunately, the targets come in disjointedly, so we need to create a
        # cache for each ID. There are better ways to do this that should be
        # explored in the future.
        for ix, id in enumerate(list(ids)):
            if id in history:
                history[id].append(data[ix, :])
                if len(history[id]) > max_seq_len:
                    history[id] = history[id][1:]
            else:
                history[id] = [data[ix, :]]

        # Prepare the batch
        batch = []
        poses = []
        for id in ids:
            datapoint = history[id]
        
            if len(datapoint) < max_seq_len:
                #print(max_seq_len-len(running_seq))
                temp_list = datapoint + [np.zeros(datapoint[0].shape) for _ in range(max_seq_len-len(datapoint))]
                input = np.array(temp_list)
                #print(input.shape)
                #input = np.rollaxis(input, 1, 0)
                pos = len(datapoint) - 1

            else:
                input = np.array(datapoint)
                #input = np.rollaxis(input, 1, 0)
                pos = max_seq_len - 1
                
            batch.append(input)
            poses.append(pos)
            
        batch = np.array(batch)
        

        
        logs = sess.run([logits], feed_dict={observation_placeholder: batch,
                                                keep_prob_placeholder: 1.0})[0]

        pred = obs.target

        pred['y'] = [logs[ix, pos]/10. for ix, pos in enumerate(poses)]
        #pred['y'] = [0 for ix, pos in enumerate(poses)]

        #pred.loc[:, 'y'] = 0.01

        #print(pred['y'][:5])

        obs, reward, done, info = env.step(pred)
        
        rewards.append(reward)
        #print(np.mean(rewards))
        #print(info["public_score_moving"])

        if done:
            print('Final score:')
            print(info["public_score"])
            break






