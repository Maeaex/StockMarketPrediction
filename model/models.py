
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score


import pandas as pd
import numpy as np

def encodeLabels(data):
    data = pd.get_dummies(data=data, columns=["week", "month"])

    return data

def trainTestSplit(data):
    # Drop features that we do not want:
    data = data.drop(columns=["Stock", "SPX", "r_diff", "r_diff_shift"])

    train = data.loc[(data['split'] == "train")]
    X_train = train.drop(columns=["split", "target"])
    y_train = train['target']

    test = data.loc[(data['split'] == "test")]
    X_test = test.drop(columns=["split", "target"])
    y_test = test['target']

    val = data.loc[(data['split'] == "validation")]
    X_val = val.drop(columns=["split", "target"])
    y_val = val['target']


    return X_train, y_train, X_test, y_test, X_val, y_val

def timeSeriesSplit(data, splits):
    data = data.drop(columns=["Stock", "SPX", "r_diff", "r_diff_shift"])
    data = data.dropna()
    data_val = data.loc[(data['split'] == "validation")]
    data_no_val = data.loc[(data['split'] != "validation")]
    X = data_no_val.drop(columns=["split", "target"])
    y = data_no_val["target"]

    X_val = data_val.drop(columns=["split", "target"])
    y_val = data_val["target"]

    tscv = TimeSeriesSplit(n_splits=splits)
    data_split = tscv.split(X)

    return tscv, X, data_split, y, X_val, y_val


def initModels(data_split):

    pipe = Pipeline(steps=[('estimator', SVC())]) # init Pipeline
    scoring = 'balanced_accuracy'

    params_grid = [{
                'estimator':[SVC(probability=True)],
                'estimator__C': [10, 100, 1000],
                'estimator__kernel': ['rbf'],                # 'poly' --> For Performance we neglect rbf for SVC
                'estimator__gamma': [0.001, 0.0001]
                },
                {
                'estimator': [DecisionTreeClassifier()],
                'estimator__max_depth': [1, 2, 5],
                'estimator__max_features': [None, "sqrt", "log2"],
                },
                {
                'estimator': [GaussianNB()],
                },

              ]
    grid = GridSearchCV(pipe, param_grid=params_grid, cv=data_split, scoring=scoring)

    return grid

def creatMlData(grid, X, X_val, y, y_val):
    #https: // scikit - learn.org / stable / auto_examples / model_selection / plot_multi_metric_evaluation.html
    y_pred = pd.DataFrame(grid.predict(X), columns=["y_pred"], index=y.index)
    y_pred_prob = pd.DataFrame(grid.predict_proba(X), columns=["y_pred_prob_pos", "y_pred_prob_neg"], index=y.index)

    score = balanced_accuracy_score(y, y_pred)

    ys = y_pred.merge(y, left_index=True, right_index=True)
    ys = ys.merge(y_pred_prob, left_index=True, right_index=True)

    comparison_column = np.where(ys["target"] == ys["y_pred"], 1, -1)
    ys["check"] = comparison_column
    comparison_column_col = np.where(ys["check"] == 1, "green", "red")
    ys["color"] = comparison_column_col

    y_pred_val = pd.DataFrame(grid.predict(X_val), columns=["y_pred"], index=y_val.index)
    y_pred_val_prob = pd.DataFrame(grid.predict_proba(X_val), columns=["y_pred_prob_pos", "y_pred_prob_neg"], index=y_val.index)

    score_val = balanced_accuracy_score(y_val, y_pred_val)

    ys_val = y_pred_val.merge(y_val, left_index=True, right_index=True)
    ys_val = ys_val.merge(y_pred_val_prob, left_index=True, right_index=True)

    comparison_column_val = np.where(ys_val["target"] == ys_val["y_pred"], 1, -1)
    ys_val["check"] = comparison_column_val
    comparison_column_col_val = np.where(ys_val["check"] == 1, "green", "red")
    ys_val["color"] = comparison_column_col_val

    bModel = grid.best_params_['estimator']


    return ys, ys_val, bModel, score, score_val

'''
{
'estimator': [AdaBoostClassifier()],
'estimator__n_estimators': [20, 50, 75],
'estimator__learning_rate': [1],
'estimator__algorithm': ["SAMME"],
'estimator__base_estimator': [SVC(kernel='rbf'), LogisticRegression()],
},

''' ## Takes to long to compute 


