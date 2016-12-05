__doc__ = """
	
	Testing util functionality

"""
import os
import sys
import imp
PROJECT_DIR = os.path.dirname(
			os.path.dirname(
				os.path.realpath(__file__)))

data_utils_path = os.path.join(PROJECT_DIR,'utils','data_utils.py')
data_utils = imp.load_source('data_utils', data_utils_path)
import data_utils
data_set = data_utils.get_data()
print data_set.keys()
df = data_set.to_df()
print df.head()