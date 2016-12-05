__doc__ = """
	
	Testing util functionality

"""
import sys
import os

PROJECT_DIR = os.path.dirname(
				os.path.dirname(
					os.path.realpath(__file__)))
sys.path.append(PROJECT_DIR)

from Two_Sigma_Financial_Modeling_Challenge.utils import data_utils

data_set = get_data()
print data_set.keys()
df = data_set.to_df()
print df.head()