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
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder, MinMaxScaler
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

    log_x = pd.DataFrame({
        'x1': train_x['f527'] - train_x['f528'],
        'x2': train_x['f274'] - train_x['f528'],
        'x3': train_x['f274'] - train_x['f527']
    })

    features_x = logistic_pipeline.fit_transform(log_x)
    # Tune the Logistic Regression
    logistic = classes.ThresholdLogisticRegression(C=1e20, threshold=0.08)
    params = {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(0, 20, num=21),
        'threshold': np.linspace(0.07, 0.11, num=10)
    }
    grid = GridSearchCV(logistic, params, scoring='f1', cv=5, n_jobs=4, verbose=50)
    grid.fit(features_x, train_y_default)
    # In [7]: grid.best_score_
    # Out[7]: 0.7293931540894103

    # In [8]: grid.best_params_
    # Out[8]: {'C': 10.0, 'penalty': 'l1', 'threshold': 0.096666666666666665}
    best_params = {'C': 10.0, 'penalty': 'l1', 'threshold': 0.096666666666666665}

    logistic = classes.ThresholdLogisticRegression(**best_params)
    logistic.fit(features_x, train_y_default)

    default_pred = logistic.predict_proba(features_x)[:, logistic.classes_].flatten()
    fpr, tpr, thresholds = roc_curve(train_y_default.values, default_pred)
    precision, recall, thresholds = precision_recall_curve(
        train_y_default.values, default_pred)
    average_precision = average_precision_score(train_y_default.values, default_pred)
    score = roc_auc_score(train_y_default.values, default_pred)

    default_pred = logistic.predict(features_x)

    # Fit the random forest
    # Also attach the default predicition features
    rf_train_x = train_x.loc[default_pred]
    rf_train_y = train_y.loc[default_pred]
    rf_train_x['x1'] = rf_train_x['f527'] - rf_train_x['f528']
    rf_train_x['x2'] = rf_train_x['f274'] - rf_train_x['f528']
    rf_train_x['x3'] = rf_train_x['f274'] - rf_train_x['f527']

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

    rf_estimator = RandomForestRegressor(n_estimators=10, oob_score=True, n_jobs=4, verbose=3, compute_importances=True)
    rf_features_x = rf_pipeline.fit_transform(rf_train_x)
    rf_estimator.fit(rf_features_x, rf_train_y)

    # Now we have two trained estimators, we predict on the test set.
    log_x_test = pd.DataFrame({
        'x1': test_x['f527'] - test_x['f528'],
        'x2': test_x['f274'] - test_x['f528'],
        'x3': test_x['f274'] - test_x['f527']
    })
    features_x_test = logistic_pipeline.transform(log_x_test)
    default_pred = logistic.predict(features_x_test)

    # Subset the test set with the predict defaults
    rf_test_x = test_x.loc[default_pred]
    rf_test_x['x1'] = rf_test_x['f527'] - rf_test_x['f528']
    rf_test_x['x2'] = rf_test_x['f274'] - rf_test_x['f528']
    rf_test_x['x3'] = rf_test_x['f274'] - rf_test_x['f527']
    rf_features_x_test = rf_pipeline.transform(rf_test_x)
    loss_pred = rf_estimator.predict(rf_features_x_test)

    # Now we have to build the composite predicted values
    preds = default_pred.astype(np.float64)
    preds[preds == 1] = loss_pred

    # Decent score of 0.65 MAE
    # 0.62 on the leaderboard with 100 trees
    score = mean_absolute_error(test_y, preds)


def staged_001_sub():
    # Make a submission
    x, y = classes.get_train_data()
    y_default = y > 0
    golden = classes.GoldenFeatures(append=False)
    impute = Imputer()
    scale = StandardScaler()
    best_params = {'C': 10.0, 'penalty': 'l1', 'threshold': 0.096666666666666665}
    logistic = classes.ThresholdLogisticRegression(**best_params)

    logistic_pipeline = Pipeline([
        ('golden', golden),
        ('impute', impute),
        ('scale', scale),
        ('logistic', logistic)
    ])

    logistic_pipeline.fit_transform(x, y_default)
    train_default_pred = logistic_pipeline.predict(x)

    rf_train_x = x.loc[train_default_pred]
    rf_train_y = y.loc[train_default_pred]

    golden_2 = classes.GoldenFeatures(append=True)
    remove_obj = classes.RemoveObjectColumns()
    remove_novar = classes.RemoveNoVarianceColumns()
    remove_unique = classes.RemoveAllUniqueColumns(threshold=0.9)
    fill_nas = Imputer()
    rf_estimator = RandomForestRegressor(n_estimators=100, n_jobs=4, verbose=3)
    rf_pipeline = Pipeline([
        ('golden', golden_2),
        ('obj', remove_obj),
        ('novar', remove_novar),
        ('unique', remove_unique),
        ('fill', fill_nas),
        ('rf', rf_estimator)
    ])
    rf_pipeline.fit(rf_train_x, rf_train_y)

    del x
    gc.collect()

    test_x = classes.get_test_data()
    test_default_pred = logistic_pipeline.predict(test_x)

    rf_test_x = test_x.loc[test_default_pred]
    loss_pred = rf_pipeline.predict(rf_test_x)

    preds = test_default_pred.astype(np.float64)
    preds[preds == 1] = loss_pred

    sub = classes.Submission(test_x.index, preds)
    sub.to_file('staged_001.csv')


def staged_002():
    """
    Trying to improve on the staged predictor

     - Try some other model besides the random forest
    """
    x, y = classes.get_train_data()
    y_default = y > 0

    train_x, test_x, \
    train_y, test_y, \
    train_y_default, test_y_default = classes.train_test_split(x, y, y_default, test_size=0.5)

    del x
    gc.collect()

    golden = classes.GoldenFeatures(append=False)
    impute = Imputer()
    scale = StandardScaler()
    best_params = {'C': 10.0, 'penalty': 'l1', 'threshold': 0.096666666666666665}
    logistic = classes.ThresholdLogisticRegression(**best_params)

    logistic_pipeline = Pipeline([
        ('golden', golden),
        ('impute', impute),
        ('scale', scale),
        ('logistic', logistic)
    ])

    logistic_pipeline.fit_transform(train_x, y_default)
    train_default_pred = logistic_pipeline.predict(train_x)

    rf_train_x = train_x.loc[train_default_pred]
    rf_train_y = train_y.loc[train_default_pred]

    golden_2 = classes.GoldenFeatures(append=True)
    remove_obj = classes.RemoveObjectColumns()
    remove_novar = classes.RemoveNoVarianceColumns()
    remove_unique = classes.RemoveAllUniqueColumns(threshold=0.9)
    fill_nas = Imputer()
    rf_estimator = RandomForestRegressor(n_estimators=100, n_jobs=4, verbose=3)
    rf_pipeline = Pipeline([
        ('golden', golden_2),
        ('obj', remove_obj),
        ('novar', remove_novar),
        ('unique', remove_unique),
        ('fill', fill_nas),
        ('rf', rf_estimator)
    ])
    rf_pipeline.fit(rf_train_x, rf_train_y)



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


def cv_for_column(x, y, c, loss=None):
    kfold = StratifiedKFold(y, n_folds=5)
    this_res = {
        'column': c,
        'auc': [],
        'avg_prec': [],
        'f1': [],
        'threshold': [],
        'mae': []
    }

    # Select the column first so we don't run into memory issues
    select = classes.ColumnSelector(cols=c)
    this_x = select.transform(x)

    for train_idx, test_idx in kfold:
        # This creates copies, so we run into memory errors
        train_x, test_x = this_x.iloc[train_idx], this_x.iloc[test_idx]
        train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]
        if loss is not None:
            train_loss, test_loss = loss.iloc[train_idx], loss.iloc[test_idx]

        impute = Imputer()
        scale = StandardScaler()
        estimator = LogisticRegression(C=1e20)
        pipeline = Pipeline([
            ('impute', impute),
            ('scale', scale),
            ('estimator', estimator)
        ])

        pipeline.fit(train_x, train_y)
        pred_test = pipeline.predict_proba(test_x)[:, estimator.classes_].flatten()
        auc = roc_auc_score(test_y.values, pred_test)
        average_precision = average_precision_score(test_y.values, pred_test)
        precision, recall, thresholds = precision_recall_curve(
            test_y.values, pred_test)
        f1s = classes.f1_scores(precision, recall)
        max_f1 = f1s.max()
        threshold = thresholds[f1s.argmax()]

        this_res['auc'].append(auc)
        this_res['avg_prec'].append(average_precision)
        this_res['f1'].append(max_f1)
        this_res['threshold'].append(threshold)
        if loss is not None:
            preds = (pred_test > threshold).astype(np.float64)
            score = mean_absolute_error(test_loss, preds)
            this_res['mae'].append(score)
    return this_res


def parallel_column_search(x, y, cs, loss=None):
    res = []
    for c in cs:
        this_res = cv_for_column(x, y, c, loss=loss)
        res.append(this_res)
    return res


def golden_features_002():
    """
    Or just looking for more features
    Some are reporting AUCs of .99 and F1s of .91 with just 2-4 features

    GridCV doesn't quite play well with this pipeline, so we'll roll our own

    This was kind of a bust, no real improvements
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
    x, y = classes.get_train_data()
    y_default = y > 0

    remove_obj = classes.RemoveObjectColumns()
    remove_novar = classes.RemoveNoVarianceColumns()
    clean = Pipeline([
        ('obj', remove_obj),
        ('novar', remove_novar),
    ])
    x = clean.fit_transform(x)

    # the bultin x.corr() seems quite slow
    # Or maybe it already got past this point and was fitting...
    # Use a subset of the rows
    corr_x = x.iloc[0:15000]
    corrs = []
    counter = 0
    # Now we want to get the pairs of columns that are highly correlated
    for i, col in enumerate(x.columns):
        if counter % 100 == 0:
            logger.info("Calculating correlations for {}".format(col))
        counter += 1
        for inner_col in x.columns[i:]:
            val = corr_x[col].corr(corr_x[inner_col])
            if val < 1 and not np.isnan(val):
                corrs.append((col, inner_col, val))

    corrs = sorted(corrs, key=lambda l: l[2], reverse=True)
    corrs = sorted(corrs, key=lambda l: l[2])

    # Actually using fewer columns seems to improve the metrics
    # If you use only one column, then most have terrible AUC and other scores
    # But the golden features seem to have substantially better AUCs
    # Or F1 scores
    # So lets try scanning the first couple hundred or so individually
    candidates = []
    for i, c in enumerate(corrs):
        if i % 10 == 0:
            logger.info("{} pairs scanned, {} candidates found".format(i, len(candidates)))

        name = c[0] + '-' + c[1]
        df = pd.DataFrame({
            # name: x[c[0]] - x[c[1]]
            name: x[c[0]] / x[c[1]]
        })
        res = cv_for_column(df, y_default, name, y)
        AUC_THRESHOLD = 0.7
        PRECISION_THRESHOLD = 0.6
        F1_THRESHOLD = 0.6
        MAE_THRESHOLD = 0.75
        averages = dict((k, sum(v) / len(v)) for k, v in res.items() if len(v) == 5)
        if averages['auc'] >= AUC_THRESHOLD or \
                averages['avg_prec'] >= PRECISION_THRESHOLD or \
                averages['f1'] >= F1_THRESHOLD or \
                averages['mae'] <= MAE_THRESHOLD:
            candidates.append(res)

    # Scanned 1100 pairs from the front with subtraction and got three candidates
    # These three get us AUC of .93/.94 and F1 of .71 - .75
    # Scanned 1000 pairs from the back with subtraction and got no candidates

    # df = pd.DataFrame(cols)
    df = pd.DataFrame({
        'x1': x['f527'] - x['f528'],
        'x2': x['f274'] - x['f528'],
        'x3': x['f274'] - x['f527']
    })
    res = cv_for_column(df, y_default, df.columns.tolist(), y)
    # Here's the current best
    res = {'auc': [0.93618096579853471,
                   0.94864059994440975,
                   0.94739596218912914,
                   0.94550080075948706,
                   0.93705845900939611],
           'avg_prec': [0.58066954118071457,
                        0.59066501806052873,
                        0.62451351853009929,
                        0.55384440264123291,
                        0.58537014840505164],
           'column': ['x1', 'x2', 'x3'],
           'f1': [0.7357723577235773,
                  0.73091718971451713,
                  0.73715124816446398,
                  0.71190383869716944,
                  0.75745366639806611],
           'mae': [0.80180137473334911,
                   0.8266413842142688,
                   0.73344394406257407,
                   0.75702839804674538,
                   0.7270184421372019],
           'threshold': [0.097046996835691582,
                         0.098343861683210645,
                         0.096169925927102232,
                         0.096602825455737562,
                         0.10014138804713153]}

    # try some categorical variables
    categoricals = ['f2', 'f4', 'f5', 'f778']
    # Programmatically generate a list of potentially categorical variables
    uniques = [len(x[xx].unique()) for xx in x.columns]
    uniques = [2 < xx < 100 for xx in uniques]
    categoricals = x.columns[uniques]

    results = []
    for cat in categoricals:
        logger.info("Trying column {}".format(cat))
        one_hot = OneHotEncoder()
        the_col = x[cat]
        # Some columns also have NANs
        # So we should impute them here
        # The imputer is somehow removing rows -- not sure what is going on
        if sum(np.isnan(the_col)) > 0:
            impute = Imputer(strategy='most_frequent', axis=0)
            imputed = impute.fit_transform(the_col.values.reshape(the_col.shape[0], 1))
            the_col = pd.Series(imputed.reshape(imputed.shape[0]), name=cat)
        # The one hot encoder only takes integers.  So manually convert to categorical integers
        if the_col.dtype == np.float64:
            convert = classes.ConvertFloatToCategory()
            the_col = convert.fit_transform(the_col)

        df = pd.DataFrame(one_hot.fit_transform(the_col.reshape((the_col.shape[0], 1))).toarray())
        res = cv_for_column(df, y_default, df.columns.tolist(), y)
        res['column'] = cat
        results.append(res)

    f1s = sorted([(xx['column'], sum(xx['f1']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)
    aucs = sorted([(xx['column'], sum(xx['auc']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)
    maes = sorted([(xx['column'], sum(xx['mae']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)

    # A few candidates to add
    top_f1s = [('f323', 0.21559999133998722),
               ('f675', 0.21338044088524746),
               ('f400', 0.21317569030181546),
               ('f676', 0.21213796918846706),
               ('f765', 0.21213796918846706),
               ('f315', 0.21138144650843232),
               ('f182', 0.19851592254698253),
               ('f192', 0.19804444167618382),
               ]

    top_aucs = [('f323', 0.61627925494709257),
                ('f675', 0.61221680691306279),
                ('f400', 0.61194975758231207),
                ('f676', 0.61150300785864187),
                ('f765', 0.61150300785864187),
                ('f315', 0.59857430298934688),
                ('f324', 0.59651102811739654),
                ('f258', 0.59247770884478279),
                ('f268', 0.59168087920153611),
                ]

    # All of these decrease the scores
    cols_to_try = {xx[0] for xx in (top_f1s + top_aucs)}

    results = []
    for col in cols_to_try:
        df = pd.DataFrame({
            'x1': x['f527'] - x['f528'],
            'x2': x['f274'] - x['f528'],
            'x3': x['f274'] - x['f527']
        })
        df[col] = x[col]
        res = cv_for_column(df, y_default, df.columns.tolist(), y)
        results.append(res)

    f1s = sorted([(xx['column'], sum(xx['f1']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)
    aucs = sorted([(xx['column'], sum(xx['auc']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)
    maes = sorted([(xx['column'], sum(xx['mae']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)

    # Object columns
    # About half seem to be actually unique
    # The other half can have as few as 6000 unique values
    # F136-138 seem to be related
    # As do f205-f208
    # Same with f275-277
    # And f336-338
    # Seem to be related by some ratio?
    # Some object columns have lots of trailing zeros, others have all non-zero digits
    # The latter does seem to be much less common
    # Maybe it's just a truncate issue
    obj_cols = [(xx, len(x[xx].unique())) for xx in x.columns if x[xx].dtype == np.object]
    obj_cols = [('f137', 6550),
                ('f138', 31152),
                ('f206', 20087),
                ('f207', 14511),
                ('f276', 6041),
                ('f277', 28710),
                ('f338', 8663),
                ('f390', 104662),
                ('f391', 104659),
                ('f419', 28965),
                ('f420', 25772),
                ('f466', 16801),
                ('f469', 86420),
                ('f472', 102913),
                ('f534', 85376),
                ('f537', 104114),
                ('f626', 104754),
                ('f627', 104751),
                ('f695', 93730),
                ('f698', 91986)]

    results = []
    for col, count in obj_cols:
        logger.info("Trying column {}".format(col))
        res = cv_for_column(x, y_default, col, y)
        res['column'] = col
        results.append(res)

    # Nothing particularly promising here
    f1s = sorted([(xx['column'], sum(xx['f1']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)
    aucs = sorted([(xx['column'], sum(xx['auc']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)
    maes = sorted([(xx['column'], sum(xx['mae']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)

    # Scan each column individually
    # No candidates
    results = []
    for i, c in enumerate(x.columns):
        if i % 100 == 0:
            logger.info("{} columns scanned".format(i))
        df = pd.DataFrame({
            'x1': x['f527'] - x['f528'],
            'x2': x['f274'] - x['f528'],
            'x3': x['f274'] - x['f527'],
            c: x[c]
        })
        res = cv_for_column(df, y_default, df.columns.tolist(), y)
        results.append(res)

    f1s = sorted([(xx['column'], sum(xx['f1']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)
    aucs = sorted([(xx['column'], sum(xx['auc']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)
    maes = sorted([(xx['column'], sum(xx['mae']) / 5.0) for xx in results], key=lambda l: l[1], reverse=True)

    # actually some really good results
    # f1 seems to have improved a lot
    top_aucs = [(['f271', 'x1', 'x2', 'x3'], 0.96596540197049252),
                (['f272', 'x1', 'x2', 'x3'], 0.95426555114415146),
                (['f2', 'x1', 'x2', 'x3'], 0.9540027948761759),
                (['f332', 'x1', 'x2', 'x3'], 0.94664438110460902),
                (['f653', 'x1', 'x2', 'x3'], 0.94637495753995871),
                (['f663', 'x1', 'x2', 'x3'], 0.94596631570439571),
                (['f662', 'x1', 'x2', 'x3'], 0.94544689548411753),
                (['f664', 'x1', 'x2', 'x3'], 0.94541053031782663),
                (['f292', 'x1', 'x2', 'x3'], 0.94473940403588386),
                (['f333', 'x1', 'x2', 'x3'], 0.94470189317437614),
                (['f273', 'x1', 'x2', 'x3'], 0.94435114467896908),
                (['f330', 'x1', 'x2', 'x3'], 0.94412664234943366),
                (['f337', 'x1', 'x2', 'x3'], 0.94408301004070661),
                (['f378', 'x1', 'x2', 'x3'], 0.94400558323914241),
                (['f297', 'x1', 'x2', 'x3'], 0.94393990069069567),
                (['f519', 'x1', 'x2', 'x3'], 0.943935287531534),
                (['f4', 'x1', 'x2', 'x3'], 0.94382428199073298),
                (['f520', 'x1', 'x2', 'x3'], 0.94380445968430104),
                (['f529', 'x1', 'x2', 'x3'], 0.94377748144604978),
                (['f617', 'x1', 'x2', 'x3'], 0.94376537514515879)]

    top_f1s = [(['f2', 'x1', 'x2', 'x3'], 0.80216366357303381),
               (['f334', 'x1', 'x2', 'x3'], 0.79656960445709735),
               (['f332', 'x1', 'x2', 'x3'], 0.77835392697909056),
               (['f339', 'x1', 'x2', 'x3'], 0.77740108790866924),
               (['f653', 'x1', 'x2', 'x3'], 0.77022939336696961),
               (['f663', 'x1', 'x2', 'x3'], 0.76845186886716743),
               (['f662', 'x1', 'x2', 'x3'], 0.76691477794164897),
               (['f271', 'x1', 'x2', 'x3'], 0.76594462112992689),
               (['f664', 'x1', 'x2', 'x3'], 0.75894687161576058),
               (['f515', 'x1', 'x2', 'x3'], 0.75654901277865938),
               (['f592', 'x1', 'x2', 'x3'], 0.75650452834130966),
               (['f591', 'x1', 'x2', 'x3'], 0.75635470273598071),
               (['f333', 'x1', 'x2', 'x3'], 0.75348589580324765),
               (['f421', 'x1', 'x2', 'x3'], 0.75099989992752414),
               (['f335', 'x1', 'x2', 'x3'], 0.74822631517846505),
               (['f415', 'x1', 'x2', 'x3'], 0.7481089001789798),
               (['f382', 'x1', 'x2', 'x3'], 0.7479311252088291),
               (['f292', 'x1', 'x2', 'x3'], 0.74695847054866993),
               (['f203', 'x1', 'x2', 'x3'], 0.74658797834693069),
               (['f336', 'x1', 'x2', 'x3'], 0.74645319772500729),
               (['f593', 'x1', 'x2', 'x3'], 0.74645298377337799),
               (['f414', 'x1', 'x2', 'x3'], 0.74467419414594205),
               (['f595', 'x1', 'x2', 'x3'], 0.74173651141144803),
               (['f416', 'x1', 'x2', 'x3'], 0.74156380168623293),
               (['f589', 'x1', 'x2', 'x3'], 0.7410922683008736),
               (['f391', 'x1', 'x2', 'x3'], 0.74104041247135177),
               (['f208', 'x1', 'x2', 'x3'], 0.7410299876414258),
               (['f627', 'x1', 'x2', 'x3'], 0.74069494845500894),
               (['f273', 'x1', 'x2', 'x3'], 0.74058748647507222),
               (['f381', 'x1', 'x2', 'x3'], 0.74013638057271114),
               (['f201', 'x1', 'x2', 'x3'], 0.74005133756653341),
               (['f390', 'x1', 'x2', 'x3'], 0.73938856841683132),
               (['f330', 'x1', 'x2', 'x3'], 0.73928581934460058),
               (['f726', 'x1', 'x2', 'x3'], 0.73893554180872645),
               (['f626', 'x1', 'x2', 'x3'], 0.738769576159654),
               (['f275', 'x1', 'x2', 'x3'], 0.73834721432124495),
               (['f535', 'x1', 'x2', 'x3'], 0.73824592184932847),
               (['f378', 'x1', 'x2', 'x3'], 0.73808081636481981),
               (['f278', 'x1', 'x2', 'x3'], 0.73797038305153018),
               (['f274', 'x1', 'x2', 'x3'], 0.73782681589431043),
               (['f527', 'x1', 'x2', 'x3'], 0.73782681589431043),
               (['f528', 'x1', 'x2', 'x3'], 0.73782681589431043),
               (['f418', 'x1', 'x2', 'x3'], 0.73782270995676624),
               (['f529', 'x1', 'x2', 'x3'], 0.73751567894378778),
               (['f534', 'x1', 'x2', 'x3'], 0.73750067493449634)]

    top_maes = [(['f416', 'x1', 'x2', 'x3'], 0.76137409103963971),
                (['f593', 'x1', 'x2', 'x3'], 0.76087162836817646),
                (['f333', 'x1', 'x2', 'x3'], 0.76083367861963502),
                (['f414', 'x1', 'x2', 'x3'], 0.76033117819170037),
                (['f335', 'x1', 'x2', 'x3'], 0.75971484649741361),
                (['f415', 'x1', 'x2', 'x3'], 0.75962952900813407),
                (['f591', 'x1', 'x2', 'x3'], 0.75934517243678257),
                (['f271', 'x1', 'x2', 'x3'], 0.75931667933587321),
                (['f664', 'x1', 'x2', 'x3'], 0.75922196453055946),
                (['f421', 'x1', 'x2', 'x3'], 0.7590037829175047),
                (['f592', 'x1', 'x2', 'x3'], 0.75834016169231344),
                (['f515', 'x1', 'x2', 'x3'], 0.75799884229219672),
                (['f662', 'x1', 'x2', 'x3'], 0.75698438299076809),
                (['f663', 'x1', 'x2', 'x3'], 0.75657669265828909),
                (['f653', 'x1', 'x2', 'x3'], 0.75561908830999314),
                (['f339', 'x1', 'x2', 'x3'], 0.75354257722549645),
                (['f332', 'x1', 'x2', 'x3'], 0.75351413896136699),
                (['f334', 'x1', 'x2', 'x3'], 0.7484795521794827),
                (['f2', 'x1', 'x2', 'x3'], 0.74550234383971203)]


def golden_feature_004():
    """
    Maybe Trying out f275 and f521 sort order
    see http://www.kaggle.com/c/loan-default-prediction/forums/t/6962/important-new-data-leakage?page=3

    Not sure if doing this right, but doesn't seem to be working
    """
    x, y = classes.get_train_data()
    y_default = y > 0

    # Strategy:
    # Iterate over the rows of 275 and 521
    # Write False to a new series if current row value of 275 matches the last
    # Write True if it doesn't match
    sort = x[['f275', 'f521']].sort(['f275', 'f521'])
    is_last = pd.DataFrame({
        'is_last': np.zeros((sort.shape[0]))
    }, index=sort.index)

    last_f275 = 0
    for i, f275, f521 in sort.itertuples():
        if f275 != last_f275:
            is_last.loc[i - 1] = 1
        last_f275 = f275

    joined = is_last.join(y)
    sum(joined.loc[joined['is_last'] == 1]['loss'] > 0)

    train_x, test_x, \
    train_y, test_y, \
    train_y_default, test_y_default = classes.train_test_split(x, y, y_default, test_size=0.2)

    """
    # Sort train_x on f275 then f521
    train_x = train_x.sort(['f275', 'f521'])
    train_x[['f275', 'f521']].iloc[1000:1050]
    train_y = train_y.loc[train_x.index]

    merged = train_x[['f275', 'f521']].join(train_y)
    x = x.sort(['f275', 'f521'])
    merged = x[['f275', 'f521']].join(y)
    # The cv split causes problems, because it can break up the sequence
    # Do any values of f275 exist in the test set?
    test = classes.get_test_data()
    test = test[['f275', 'f521']]
    joined = test.join(merged, on='f275', lsuffix='test', rsuffix='train')
    joined.loc[np.logical_not(np.isnan(joined['f275train']))]
    # Seems to be only 204 duplicates
    # But something weird about this join, the columns aren't actually equal
    # This method gives only 804 duplicates
    test = test['f275'].unique()
    merged = merged['f275'].unique()
    rows = []
    for r in merged:
        if r in test:
            rows.append(r)
    """
