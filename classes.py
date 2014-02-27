from __future__ import division
import logging
import os
import pandas as pd
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
        n_unique = [len(x.unique()) for n, x in X.iteritems()]
        self.mask_ = [x < threshold for x in n_unique]
        return self

    def transform(self, X):
        return X.loc[:, self.mask_]


class RemoveNoVarianceColumns(BaseEstimator, TransformerMixin):
    """
    Remove all columns that have no variance (1 unique value)
    """
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        n_unique = [len(x.unique()) for n, x in X.iteritems()]
        self.mask_ = [x != 1 for x in n_unique]
        return self

    def transform(self, X):
        return X.loc[:, self.mask_]
