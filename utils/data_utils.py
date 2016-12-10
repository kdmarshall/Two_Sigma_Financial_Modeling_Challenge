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
	def __init__(self, path):
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
				print "Slice is out of range. Taking last batch_size slice"
				sample_range = (num_samples - batch_size, num_samples)
			else:
				sample_range = (slice_index*batch_size, (slice_index+1)*batch_size)
			samples = self.df[sample_range[0] : sample_range[1]]
		samples_matrix = np.array(samples.as_matrix(columns=columns)) if columns else np.array(samples.as_matrix())
		return samples_matrix
		
