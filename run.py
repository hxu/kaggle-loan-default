from __future__ import division
import gc
import itertools
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing
from sklearn.externals.joblib import Parallel, delayed
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, mean_absolute_error, \
    average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
import classes
from classes import plot
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
    train_y_default, test_y_default = classes.train_test_split(x, y, y_default, test_size=0.2)

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
    # This will give us a set where about 1/3 of the observations are true positives
    default_pred = logistic.predict_proba(features_x)[:, logistic.classes_].flatten()
    fpr, tpr, thresholds = roc_curve(train_y_default.values, default_pred)
    precision, recall, thresholds = precision_recall_curve(
        train_y_default.values, default_pred)
    average_precision = average_precision_score(train_y_default.values, default_pred)
    score = roc_auc_score(train_y_default.values, default_pred)
    threshold = classes.get_threshold(fpr, tpr, thresholds)
    log_mask = default_pred > threshold

    # Fit the random forest
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

    rf_estimator = RandomForestRegressor(n_estimators=10, oob_score=True, n_jobs=4, verbose=3)
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

    # This is a pretty terrible MAE score of 2.388 when training RF on true positives only
    # MAE goes to 1.6 or so when training on predicted positives
    score = mean_absolute_error(test_y, preds)

    # what happens if we don't use the RF and just use the average of the defaults?
    # MAE of 1.2 or so
    # Issue seems to be still picking up too many false positives
    mean = rf_train_y.mean()
    preds = default_pred.astype(np.float64)
    preds[preds == 1] = mean
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
    # mask = y > 0
    # x = x.loc[mask]
    # y = y.loc[mask]

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
    mask = train_y > 0
    features_x = pipeline.fit_transform(train_x.loc[mask])
    estimator.fit(features_x, train_y.loc[mask])

    mask = test_y > 0
    features_x_test = pipeline.transform(test_x.loc[mask])
    pred = estimator.predict(features_x_test)
    # 5.40844 on 4892 samples (2-fold split)
    score = mean_absolute_error(test_y, pred)

    # Ceiling analysis - assume we can get get a classifier that is 100% accurate, what would the MAE be?
    all_pred = test_y.copy().astype(np.float64)
    all_pred.loc[all_pred > 0] = pred
    # 0.50, so actually a really good score
    # I guess the strategy should be to substantially improve the quality of the classifer
    score = mean_absolute_error(test_y, all_pred)


def golden_features_001():
    """
    http://www.kaggle.com/c/loan-default-prediction/forums/t/7115/golden-features
    """
    x, y = classes.get_train_data()
    # x = x[['f527', 'f528']]
    # Adding f247 makes the result worse
    # x = x[['f527', 'f528', 'f247']]
    # Diff doesn't improve any

    # Using these diffs takes teh f1 score to 0.73!
    x = pd.DataFrame({
        'x1': x['f527'] - x['f528'],
        'x2': x['f528'] - x['f274'],
        # 'x3': x['f527'] - x['f274']
    })
    y_default = y > 0

    train_x, test_x, \
    train_y, test_y, \
    train_y_default, test_y_default = classes.train_test_split(x, y, y_default, test_size=0.2)

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
    pred_y = estimator.predict_proba(features_x_test)[:, estimator.classes_].flatten()

    # Precision: tp / (tp + fp)
    # Recall: tp / (tp + fn)
    score = roc_auc_score(test_y_default.values, pred_y)
    precision, recall, thresholds = precision_recall_curve(
        test_y_default.values, pred_y)
    # so the tpr is the # of true positives / total positives
    # fpr must be # of false positives / negatives
    fpr, tpr, thresholds = roc_curve(test_y_default.values, pred_y)
    f1s = classes.f1_scores(precision, recall)
    average_precision = average_precision_score(test_y_default.values, pred_y)
    threshold = classes.get_threshold(fpr, tpr, thresholds)

    # df = pd.DataFrame({"actuals": test_y, "predicted": pred_y.flatten() > threshold})
    # predicted_defaults = df[df['predicted']]
    # This gets an AUC of .92 or so
    # Average precision of .41
    # Using the diffs gets 0.58 of average precision
    # classes.plot_roc(fpr, tpr)

    # Using mean of training Ys doesn't help score
    preds = pred_y > threshold
    preds = preds.astype(np.float64)
    # Score of 0.78
    score = mean_absolute_error(test_y, preds)


def cv_for_column(x, y, c):
    logger.info("Testing column {}".format(c))
    kfold = StratifiedKFold(y, n_folds=5)
    this_res = {
        'column': c,
        'auc': [],
        'avg_prec': [],
        'f1': [],
        'threshold': []
    }

    # Select the column first so we don't run into memory issues
    select = classes.ColumnSelector(cols=c)
    this_x = select.transform(x)

    for train_idx, test_idx in kfold:
        # This creates copies, so we run into memory errors
        train_x, test_x = this_x.iloc[train_idx], this_x.iloc[test_idx]
        train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]

        impute = Imputer()
        scale = StandardScaler()
        estimator = LogisticRegression(C=1e20)
        pipeline = Pipeline([
            ('impute', impute),
            ('scale', scale),
            ('estimator', estimator)
        ])

        pipeline.fit(train_x, train_y)
        pred_y = pipeline.predict_proba(test_x)[:, estimator.classes_].flatten()
        auc = roc_auc_score(test_y.values, pred_y)
        average_precision = average_precision_score(test_y.values, pred_y)
        precision, recall, thresholds = precision_recall_curve(
            test_y.values, pred_y)
        f1s = classes.f1_scores(precision, recall)
        max_f1 = f1s.max()
        threshold = thresholds[f1s.argmax()]
        this_res['auc'].append(auc)
        this_res['avg_prec'].append(average_precision)
        this_res['f1'].append(max_f1)
        this_res['threshold'].append(threshold)
    return this_res


def parallel_column_search(x, y, cs):
    res = []
    for c in cs:
        this_res = cv_for_column(x, y, c)
        res.append(this_res)
    return res


def golden_features_002():
    """
    Maybe Trying out f275 and f521 sort order
    see http://www.kaggle.com/c/loan-default-prediction/forums/t/6962/important-new-data-leakage?page=3

    Or just looking for more features
    Some are reporting AUCs of .99 and F1s of .91 with just 2-4 features

    GridCV doesn't quite play well with this pipeline, so we'll roll our own
    """

    x, y = classes.get_train_data()
    y_default = y > 0

    remove_obj = classes.RemoveObjectColumns()
    remove_novar = classes.RemoveNoVarianceColumns()
    remove_unique = classes.RemoveAllUniqueColumns(threshold=0.98)
    clean = Pipeline([
        ('obj', remove_obj),
        ('novar', remove_novar),
        ('unique', remove_unique),
    ])
    x = clean.fit_transform(x)

    # Columns we want to search over
    cols = x.columns.tolist()
    chunks = list(classes.chunks(cols, 4))

    res = Parallel(n_jobs=4, verbose=3)(
        delayed(parallel_column_search)(x, y_default, cs) for cs in chunks
    )

    # Get the 50 best columns from each
    best_columns = set()

    # Top f1 is about .219
    f1s = sorted([(r['column'], sum(r['f1']) / 5.0) for r in res], key=lambda l: l[1], reverse=True)
    # Top auc is .634
    auc = sorted([(r['column'], sum(r['auc']) / 5.0) for r in res], key=lambda l: l[1], reverse=True)
    # Top precision is 0.546 -- the best performers perform much better than the next best
    prec = sorted([(r['column'], sum(r['avg_prec']) / 5.0) for r in res], key=lambda l: l[1], reverse=True)
    # Basically 0.8 - 0.11
    threshold = sorted([(r['column'], sum(r['threshold']) / 5.0) for r in res], key=lambda l: l[1], reverse=True)

    # Gives us 81, meaning there's an overlap
    for metric in [f1s, auc, prec]:
        for feat in metric[0:50]:
            best_columns.add(feat[0])

    # Doesn't seem to include the golden features that others have reported
    # One of the golden features is also getting filtered out by the remove_unique
    # This is because we're also removing legitimate floats
    best_columns = {'f129', 'f14', 'f142', 'f15', 'f17',
                    'f182', 'f188', 'f191', 'f192', 'f198',
                    'f20', 'f201', 'f21', 'f22', 'f24',
                    'f25', 'f26', 'f262', 'f268', 'f281',
                    'f282', 'f283', 'f305', 'f31', 'f314',
                    'f315', 'f32', 'f321', 'f322', 'f323',
                    'f324', 'f329', 'f332', 'f333', 'f351',
                    'f352', 'f376', 'f377', 'f395', 'f396', 'f397',
                    'f398', 'f399', 'f4', 'f400', 'f402', 'f404', 'f405',
                    'f406', 'f424', 'f443', 'f46', 'f50', 'f517',
                    'f56', 'f595', 'f60', 'f604', 'f629', 'f63', 'f630',
                    'f64', 'f648', 'f649', 'f666', 'f669', 'f671',
                    'f675', 'f676', 'f678', 'f679', 'f723', 'f724',
                    'f725', 'f763', 'f765', 'f766', 'f767', 'f768',
                    'f776', 'f777'}

    # check the full results of each column
    selected_res = [r for r in res if r['column'] in best_columns]

    # Now lets do pairs or triples of each of the best
    # this will take a looong time, 85,000 possible combinations of 3
    cols = itertools.chain(itertools.combinations(best_columns, 2), itertools.combinations(best_columns, 3))
    res = []

    for c in cols:
        this_res = cv_for_column(x, y_default, c)
        res.append(this_res)


def golden_features_003():
    """
    Some have suggested to look for differences between highly correlated columns
    Then optimize around MAE
    """
