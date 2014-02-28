import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
import classes
from classes import logger
from constants import *
import matplotlib.pylab as pl


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


def logistic_001():
    # Logistic regression will automatically oversample the rarer class, maybe
    # this will perform better than RF
    # However, it requires that the features be scaled and centered,
    # which means that we'll need to account for categorical variables as well
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
    pipeline = Pipeline([
        ('obj', remove_obj),
        ('novar', remove_novar),
        ('unique', remove_unique),
        ('fill', fill_nas),
    ])

    features_x = pipeline.fit_transform(train_x)
    one_hot = classes.CategoricalExpansion(threshold=30)


def golden_features_001():
    """
    http://www.kaggle.com/c/loan-default-prediction/forums/t/7115/golden-features
    """
    x, y = classes.get_train_data()
    x = x[['f527', 'f528', 'f247']]
    y_default = y > 0

    train_x, test_x, \
    train_y, test_y, \
    train_y_default, test_y_default = classes.train_test_split(x, y, y_default, test_size=0.5)

    del x
    gc.collect()

    impute = Imputer()
    scale = StandardScaler()

    pipeline = Pipeline([
        ('impute', impute),
        ('scale', scale),
    ])

    features_x = pipeline.fit_transform(train_x)
    estimator = LogisticRegression(C=1e20)
    estimator.fit(features_x, train_y_default)

    features_x_test = pipeline.transform(test_x)

    # Get only positive probabilities
    pred_y = estimator.predict_proba(features_x_test)[:, estimator.classes_]

    # Precision: tp / (tp + fp)
    # Recall: tp / (tp + fn)
    score = roc_auc_score(test_y_default.values, pred_y)
    precision, recall, thresholds = precision_recall_curve(
        test_y_default.values, pred_y)
    fpr, tpr, thresholds = roc_curve(test_y_default.values, pred_y)


    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % score)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()