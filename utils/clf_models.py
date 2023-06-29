from sklearn.model_selection import LeavePOut, LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time
import pandas as pd
import numpy as np
from sklearn.base import clone


def get_models():
    """
    The function returns all the model for classifier the data
    :return: dictionary map model name to its class
    """
    return {'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'RandomForest': RandomForestClassifier(),
            'LogisticsRegression': LogisticRegression(),
            'NB': GaussianNB()}


def cv_djustment(sample_size):
    """
    Adjust the Cross-Validation method to the data
    :param sample_size: number of sample
    :return: cv methods and the name
    """
    if sample_size < 50:
        return 'Leave-pair-out', LeavePOut(2)
    elif 50 <= sample_size < 100:
        return 'LOOCV', LeaveOneOut()
    elif 100 <= sample_size < 1000:
        return '10-Fold-CV', KFold(n_splits=10, random_state=100, shuffle=True)
    else:
        return '5-Fold-CV', KFold(n_splits=5, random_state=100, shuffle=True)


def get_scores(y_test, y_pred, y_score, multi_class):
    """
    calculate metric for scoring the ML model. for binary class and multi-class
    :param y_test:
    :param y_pred:
    :param y_score:
    :param multi_class:
    :return:
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    scoring_dict = dict()
    if multi_class:
        scoring_dict['ACC'] = accuracy_score(y_test.argmax(axis=-1), y_pred.argmax(axis=-1))
        scoring_dict['MCC'] = matthews_corrcoef(y_test.argmax(axis=-1), y_pred.argmax(axis=-1))
        fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
        scoring_dict['AUC'] = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        scoring_dict['PR-AUC'] = auc(recall, precision)
    else:
        scoring_dict['ACC'] = accuracy_score(y_test, y_pred)
        scoring_dict['MCC'] = matthews_corrcoef(y_test, y_pred)
        scoring_dict['AUC'] = roc_auc_score(y_test, y_score[:, 1])
        precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])
        scoring_dict['PR-AUC'] = auc(recall,precision)
    return scoring_dict


def evaluate_model(model, X, y, cv_name, cv, multi_class=False):
    """
    Build each model for each fold
    :param model: ML model
    :param X: X data
    :param y: the target variable
    :param cv_name: cv type
    :param cv: CV method
    :param multi_class: boolean
    :return:
    """
    fit_time, pred_time, split = 0, 0, 1
    pred, pred_prob, y_test_all = [], [], []

    if multi_class:
        y = pd.get_dummies(y).to_numpy()
        model = OneVsRestClassifier(model)

    for train, test in cv.split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        y_test_all.extend(y_test.tolist())
        model = clone(model)

        # fit the model
        if split == 1:
            start = time.time()
        model.fit(X_train, y_train)
        if split == 1:
            fit_time = time.time() - start
            start = time.time()

        # predict
        y_pred = model.predict(X_test)
        if split == 1:
            pred_time = time.time() - start
        pred.extend(y_pred.tolist())

        # score
        if multi_class:
            if hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                y_score = model.predict_proba(X_test)
            pred_prob.extend(y_score.tolist())
        else:
            pred_prob.extend((model.predict_proba(X_test)).tolist())
        split += 1

    # results conclusion
    scoring_dict = get_scores(y_test_all, pred, pred_prob, multi_class)
    scoring_dict['fit_time'] = fit_time
    scoring_dict['pred_time'] = pred_time
    scoring_dict['cv'] = cv_name
    scoring_dict['folds'] = split

    return scoring_dict
