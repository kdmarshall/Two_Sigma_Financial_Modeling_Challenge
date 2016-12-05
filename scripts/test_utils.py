__doc__ = """
	
	Testing util functionality

"""
from Two_Sigma_Financial_Modeling_Challenge.utils import data_utils

data_set = get_data()
print data_set.keys()
df = data_set.to_df()
print df.head()