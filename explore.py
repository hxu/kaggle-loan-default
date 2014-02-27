from __future__ import division
import classes
from constants import *
import numpy as np
import pandas as pd

train_x, train_y = classes.get_train_data()

# Train x is of type
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 105471 entries, 1 to 105471
# Columns: 769 entries, f1 to f778
# dtypes: float64(652), int64(97), object(20)

# the int64s are probably good candidates for categorical variables
# The object data types turn out to be really long numbers.  Maybe account numbers or something?
object_types_mask = train_x.dtypes == np.object
object_types = train_x.loc[:, object_types_mask]
object_types.iloc[0:5]

# Possible leakage via the object types numbers?

# Actually from the description we can see that some of these columns may be useful (the unique count is not very close to the total count)
desc = object_types.describe()

# IDEA - filter out columns where unique ~ count

# Find out how many unique ints there are in each int64 column
int_types_mask = train_x.dtypes == np.int64
int_types = train_x.loc[:, int_types_mask]
desc = int_types.describe()
# Some columns have 0 standard deviation
# IDEA - filter out columns where SD ~ 0

# most
unique_ints = [len(x.unique()) for n, x in int_types.iteritems()]


# Looks like about half the rows have NAs in them.  Can't drop rows, otherwise lose too many observations
# Are NAs possibly correlated with defaults?
# 431 columns have NANs, so can't drop columns either
# All the columns appear to be float columns -- can just interpolate maybe?
pd.DataFrame
