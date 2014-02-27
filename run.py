import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import classes
from classes import logger
from constants import *


def rf_001():
    """
    Staged predictions - first predict whether default or not, then predict amount of default
    """
    x, y = classes.get_train_data()
    y_default = y > 0

    train_x, test_x, \
        train_y, test_y, \
        train_y_default, test_y_default = classes.train_test_split(x, y, y_default, test_size=0.5)

    del x
    gc.collect()

    remove_obj = classes.RemoveObjectColumns()
    remove_novar = classes.RemoveNoVarianceColumns()
    remove_unique = classes.RemoveAllUniqueColumns(threshold=0.9)
    fill_nas = classes.FillNAsWithMean()
    default_predictor = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4, verbose=3)
    pipeline = Pipeline([
        ('obj', remove_obj),
        ('novar', remove_novar),
        ('unique', remove_unique),
        ('fill', fill_nas),
    ])

    # Leave RF out of the pipeline for finer tuning
    res = pipeline.fit_transform(train_x)
    default_predictor.fit(res, train_y_default)
    res_test = pipeline.fit_transform(test_x)
    pred = default_predictor.predict(res_test)

