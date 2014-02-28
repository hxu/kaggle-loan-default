from __future__ import division
from itertools import chain
import logging
import os
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from constants import *
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


logger = logging.getLogger('loan_default')
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
# Log to file
logfile = logging.FileHandler('run.log')
logfile.setLevel(logging.DEBUG)
logfile.setFormatter(log_formatter)
# Log to console
logstream = logging.StreamHandler()
logstream.setLevel(logging.INFO)
logstream.setFormatter(log_formatter)

logger.addHandler(logfile)
logger.addHandler(logstream)


def get_train_data():
    # Returns both the x and the y
    data = pd.read_csv(TRAIN_DATA_FILE, na_values='NA', index_col='id')
    return data.iloc[:, 0:-1], data['loss']


def get_train_x():
    # Loss column is the last column, so we remove it
    return pd.read_csv(TRAIN_DATA_FILE, na_values='NA', index_col='id').iloc[:, 0:-1]


def get_train_y():
    return pd.read_csv(TRAIN_DATA_FILE, na_values='NA', index_col='id', usecols=['id', 'loss'])


class Submission(object):
    """
    Utility function that takes care of some common tasks relating to submissiosn files
    """
    submission_format = ['%i', '%i']

    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_file(filename):
        """
        Load a submission from a csv file
        """
        pass

    def to_file(self, filename):
        """
        Output a submission to a file
        """
        outpath = os.path.join(SUBMISSION_PATH, filename)
        logger.info("Saving solutions to file {}".format(outpath))


class RemoveObjectColumns(BaseEstimator, TransformerMixin):
    """
    Given a df, remove all columns of type object
    """
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        self.mask_ = X.dtypes != np.object
        return self

    def transform(self, X):
        logger.info("Removing object columns from:")
        logger.info(X)
        return X.loc[:, self.mask_]


class RemoveAllUniqueColumns(BaseEstimator, TransformerMixin):
    """
    Remove all columns where unique ~= n_rows.

    Arguments:
    ==========
    threshold: float
        Specifies how aggressively to remove columns.  If threshold is 0.9, then
        any column where n_unique >= 0.9 * n_row is removed
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X=None, y=None):
        n_rows = X.shape[0]
        threshold = n_rows * self.threshold
        # Why is this so much faster than x.nunique()??
        n_unique = [len(x.unique()) for n, x in X.iteritems()]
        # n_unique = [x.nunique() for n, x in X.iteritems()]
        self.mask_ = [x < threshold for x in n_unique]
        return self

    def transform(self, X):
        logger.info("Removing columns with {}% unique values".format(self.threshold))
        logger.info(X)
        return X.loc[:, self.mask_]


class RemoveNoVarianceColumns(BaseEstimator, TransformerMixin):
    """
    Remove all columns that have no variance (1 unique value)
    """
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        n_unique = [len(x.unique()) for n, x in X.iteritems()]
        # n_unique = [x.nunique() for n, x in X.iteritems()]
        self.mask_ = [x != 1 for x in n_unique]
        return self

    def transform(self, X):
        logger.info("Removing columns with no variance")
        logger.info(X)
        return X.loc[:, self.mask_]


class FillNAsWithMean(BaseEstimator, TransformerMixin):
    """
    DEPRECATED: Use sklearn.preprocessing.Imputer

    Fill NAs with the mean of the column in place

    Pandas fill methods are OK if we want the same value for all NAs,
    or if we are doing a timeseries, but they are not quite appropriate for this dataset
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        desc = X.describe()
        self.means_ = desc.loc['mean']
        return self

    def transform(self, X):
        logger.info("Filling NAs with column means")
        logger.info(X)
        desc = X.describe()
        # Test set could have NAs in columns that didn't have NAs in train est
        nas = desc.loc['count'] < X.shape[0]
        for i, colname in enumerate(X.columns):
            if nas[i]:
                X.loc[pd.isnull(X[colname]), colname] = self.means_[i]
        return X


class CategoricalExpansion(BaseEstimator, TransformerMixin):
    """
    Uses one hot encoder to expand categorical columns
    Don't use this in a pipeline

    Arguments:
    =========
    threshold: int
        The maximum number of unique values that a column can have
        for it to be considered categorical

    Returns:
    ========
    Sparse matrix of expanded column.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        uniques = [(len(x.unique()), x.dtype.kind) for n, x in X.iteritems()]
        self.mask_ = [(x[0] < self.threshold and x[1] == 'i') for x in uniques]
        self.encoder_ = OneHotEncoder()
        self.encoder_.fit(X.loc[:, self.mask_])
        return self

    def transform(self, X):
        return self.encoder_.transform(X.loc[:, self.mask_])


def train_test_split(*arrays, **options):
    """
    Adapted split utility for pandas data frames
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    options['sparse_format'] = 'csr'

    if test_size is None and train_size is None:
        test_size = 0.25

    n_samples = arrays[0].shape[0]
    cv = ShuffleSplit(n_samples, test_size=test_size,
                      train_size=train_size,
                      random_state=random_state)

    train, test = next(iter(cv))
    return list(chain.from_iterable((a.iloc[train], a.iloc[test]) for a in arrays))

