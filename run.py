import gc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
import classes
import numpy as np
import pandas as pd
from classes import logger
from constants import *
import matplotlib.pylab as pl


def staged_001():
    """
    Staged predictions - first predict whether default or not, then predict amount of default
    """
    # First we predict the defaults
    LOG_COLUMNS = ['f527', 'f528']
    x, y = classes.get_train_data()
    y_default = y > 0

    train_x, test_x, \
    train_y, test_y, \
    train_y_default, test_y_default = classes.train_test_split(x, y, y_default, test_size=0.5)

    del x
    gc.collect()

    # Fit the logistic regression
    impute = Imputer()
    scale = StandardScaler()

    logistic_pipeline = Pipeline([
        ('impute', impute),
        ('scale', scale),
    ])

    features_x = logistic_pipeline.fit_transform(train_x[LOG_COLUMNS])
    logistic = LogisticRegression(C=1e20)
    logistic.fit(features_x, train_y_default)

    # Now we have to pick a threshold
    # We'll pick a threshold that maximizes tpr - fpr

    # It seems that the logistic regression is not actually that good.
    # Even with the high AUC, the recall /precision seem to be quite bad
    default_pred = logistic.predict_proba(features_x)[:, logistic.classes_].flatten()
    fpr, tpr, thresholds = roc_curve(train_y_default.values, default_pred)
    precision, recall, thresholds = precision_recall_curve(
        train_y_default.values, default_pred)
    score = roc_auc_score(train_y_default.values, default_pred, average="weighted")
    threshold = classes.get_threshold(fpr, tpr, thresholds)
    log_mask = default_pred > threshold

    # Fit the random forest
    # We'll use the actual defaults instead of the ones that the logistic regression tells us to use
    # Not sure if this is correct or not -- maybe should use the samples that the logistic regression predicts are defaults
    mask = train_y > 0
    rf_train_x = train_x.loc[log_mask]
    rf_train_y = train_y.loc[log_mask]

    remove_obj = classes.RemoveObjectColumns()
    remove_novar = classes.RemoveNoVarianceColumns()
    remove_unique = classes.RemoveAllUniqueColumns(threshold=0.9)
    fill_nas = Imputer()
    rf_pipeline = Pipeline([
        ('obj', remove_obj),
        ('novar', remove_novar),
        ('unique', remove_unique),
        ('fill', fill_nas),
    ])

    rf_estimator = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=4, verbose=3)
    rf_features_x = rf_pipeline.fit_transform(rf_train_x)
    rf_estimator.fit(rf_features_x, rf_train_y)

    # Now we have two trained estimators, we predict on the test set.
    features_x_test = logistic_pipeline.transform(test_x[LOG_COLUMNS])
    default_pred = logistic.predict_proba(features_x_test)[:, logistic.classes_].flatten()
    default_pred = default_pred > threshold

    # Subset the test set with the predict defaults
    rf_test_x = test_x.loc[default_pred]
    rf_features_x_test = rf_pipeline.transform(rf_test_x)
    loss_pred = rf_estimator.predict(rf_features_x_test)

    # Now we have to build the composite predicted values
    preds = default_pred.astype(np.float64)
    preds[preds == 1] = loss_pred

    # This is a pretty terrible MAE score of 2.388
    score = mean_absolute_error(test_y, preds)


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
    fill_nas = Imputer()
    pipeline = Pipeline([
        ('obj', remove_obj),
        ('novar', remove_novar),
        ('unique', remove_unique),
        # Filling NAs will convert to ndarray
        # But we need dframe for categorical expansion
        # ('fill', fill_nas),
    ])

    features_x = pipeline.fit_transform(train_x)
    one_hot = classes.CategoricalExpansion(threshold=30)
    categoricals = one_hot.fit_transform(features_x)
    # Remove the categorical features from the dataset
    mask = [not x for x in one_hot.mask_]
    features_x = features_x.loc[:, mask]
    features_x = fill_nas.fit_transform(features_x)
    # now stack
    features_x = np.hstack([features_x, categoricals])
    # Standardize
    scale = StandardScaler()
    features_x = scale.fit_transform(features_x)

    estimator = LogisticRegression(C=1e20)
    # This doesn't finish in a reasonable amount of time.  Too many dimensions
    estimator.fit(features_x, train_y_default)

    # Now generate the test set
    features_x_test = pipeline.transform(test_x)
    categoricals_test = one_hot.transform(features_x_test)
    features_x_test = features_x.loc[:, mask]
    features_x_test = fill_nas.fit_transform(features_x_test)
    features_x_test = np.hstack([features_x_test, categoricals_test])
    features_x_test = scale.transform(features_x_test)

    # Get only positive probabilities
    pred_y = estimator.predict_proba(features_x_test)[:, estimator.classes_]


def loss_001():
    """
    Predicting only the losses
    """
    x, y = classes.get_train_data()
    # get only the rows with losses
    mask = y > 0
    x = x.loc[mask]
    y = y.loc[mask]

    train_x, test_x, \
        train_y, test_y, = classes.train_test_split(x, y, test_size=0.5)

    del x
    gc.collect()

    remove_obj = classes.RemoveObjectColumns()
    remove_novar = classes.RemoveNoVarianceColumns()
    remove_unique = classes.RemoveAllUniqueColumns(threshold=0.9)
    fill_nas = Imputer()
    pipeline = Pipeline([
        ('obj', remove_obj),
        ('novar', remove_novar),
        ('unique', remove_unique),
        ('fill', fill_nas),
    ])

    estimator = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=4, verbose=3)
    features_x = pipeline.fit_transform(train_x)
    estimator.fit(features_x, train_y)

    features_x_test = pipeline.transform(test_x)
    pred = estimator.predict(features_x_test)
    # 5.40844 on 4892 samples (2-fold split)
    score = mean_absolute_error(test_y, pred)


def golden_features_001():
    """
    http://www.kaggle.com/c/loan-default-prediction/forums/t/7115/golden-features
    """
    x, y = classes.get_train_data()
    x = x[['f527', 'f528']]
    # Adding f247 makes the result worse
    # x = x[['f527', 'f528', 'f247']]
    # Should also try diffing these columns
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
    # so the tpr is the # of true positives / total positives
    # fpr must be # of false positives / negatives
    fpr, tpr, thresholds = roc_curve(test_y_default.values, pred_y)
    threshold = classes.get_threshold(fpr, tpr, thresholds)

    df = pd.DataFrame({"actuals": test_y, "predicted": pred_y.flatten() > threshold})
    predicted_defaults = df[df['predicted']]

    # This gets an AUC of .92 or so
    classes.plot_roc(fpr, tpr)
