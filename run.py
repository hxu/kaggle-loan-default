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
    train_x, train_y = classes.get_train_data()

    remove_obj = classes.RemoveObjectColumns()
    remove_novar = classes.RemoveNoVarianceColumns()
    remove_unique = classes.RemoveAllUniqueColumns(threshold=0.9)
    convert_categorical = classes.ConvertToCategorical(max_cat=20)
    scale = StandardScaler()
    pipeline = Pipeline([
        ('obj', remove_obj),
        ('novar', remove_novar),
        ('unique', remove_unique),
        ('categorical', convert_categorical)
    ])

    train_x = pipeline.transform(train_x)
    train_y_default = train_y > 0

    default_predictor = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4, verbose=3)
