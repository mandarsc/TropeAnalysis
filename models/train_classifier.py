from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


def train_eval_classifiers(X_train, X_test, y_train: np.array, y_test: np.array, classifier: str = 'xgb', multi_output: bool = False, n_estimators: int = 100) -> Tuple[float, float]:
    """
    This function trains a Random Forest or Xgboost classifier, evaluates it and returns AUC and Average precision scores.
    Args:
        X_train: Numpy array containing train set
        X_test: Numpy array containing test set
        y_train: Numpy array containing labels for train set
        y_test: Numpy array containing labels for test set
        classifier: String containing name of classifier to train. Should be one of two values: 'rf' or 'xgb'
        multi_output: Boolean containing whether to train multi_output classifier
        n_estimators: Integer specifying number of decision trees to train classifier

    Returns:
        Tuple[float, float]: Tuple containing AUC score and Average precision score
    """
    if classifier == 'xgb':
        _fit = XGBClassifier(eval_metric='logloss', use_label_encoder=False, n_estimators=n_estimators)
    elif classifier == 'rf':
        _fit = RandomForestClassifier(n_estimators=n_estimators)

    if multi_output:
        classifier = MultiOutputClassifier(estimator=_fit)
        classifier.fit(X_train, y_train)
        y_hat = mutli_out_classifier.predict(X_test)
        auc_score = np.zeros(len(y_test[0]))
        avg_precision = np.zeros(len(y_test[0]))
        for i in range(len(y_test[0])):
            auc_score[i] = roc_auc_score(y_test[:, i],y_hat[:, i])
            avg_precision[i] = average_precision_score(y_test[:, i],y_hat[:, i])
    else:
        _fit.fit(X_train, y_train)
        y_hat = _fit.predict_proba(X_test)
        auc_score = roc_auc_score(y_test,y_hat[:, 1])
        avg_precision = average_precision_score(y_test,y_hat[:, 1])

    return auc_score, avg_precision
