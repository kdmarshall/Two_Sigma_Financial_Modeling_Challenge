__doc__ = """
    
    Various data utilities.

"""
####################################################################
# Packages
####################################################################
import os
import h5py
import numpy as np
import pandas as pd

####################################################################
# Globals/Constants
####################################################################
PROJECT_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'train.h5')

####################################################################
# Functions
####################################################################
def get_data(path=None):
    if path:
        data_set = DataSet(path)
    else:
        data_set = DataSet(TRAIN_DATA_FILE)
    return data_set

####################################################################
# Classes
####################################################################

class DataSet(object):
    """class for dataset processing"""
    def __init__(self, path=TRAIN_DATA_FILE):
        self.path = path
        self.data_dict = self._get_data_dict()
        self.df = self._get_df()

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
        
        means = []
        stds = []
        
        # Assuming column order remains consistent throughout the class
        for col in df.columns:
            if col not in ['y', 'timestamp', 'index', 'id']:
                data = df[col].dropna().as_matrix()
                means.append(np.mean(data))
                stds.append(np.std(data))
    
        col_means = np.array(means)
        col_stds = np.array(stds)

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
        examples = (examples - col_means)/col_stds
        
        # TODO: Supply these outside the function later: col_means, col_stds

        return examples, targets, weights

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
        
        valid_len = 300 # Hardcoded for now

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







