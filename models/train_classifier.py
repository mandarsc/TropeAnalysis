from collections import defaultdict
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_curve, roc_auc_score
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
        Dict[str, Any]: Dictinoary containing AUC score, Average precision score and f1 score metrics.
    """
    if classifier == 'xgb':
        _fit = XGBClassifier(eval_metric='logloss', use_label_encoder=False, n_estimators=n_estimators)
    elif classifier == 'rf':
        _fit = RandomForestClassifier(n_estimators=n_estimators)

    metrics_dict = defaultdict(str)
    
    if multi_output:
        metrics_dict['AP'] = np.zeros(len(y_test[0]))
        metrics_dict['AUC'] = np.zeros(len(y_test[0]))
        classifier = MultiOutputClassifier(estimator=_fit)
        classifier.fit(X_train, y_train)
        y_hat = mutli_out_classifier.predict(X_test)

        for i in range(len(y_test[0])):
            metrics_dict['AUC'][i] = roc_auc_score(y_test[:, i],y_hat[:, i])
            metrics_dict['AP'][i] = average_precision_score(y_test[:, i],y_hat[:, i])
            # metrics_dict['F1'][i] = f1_score(y_test[:, i],y_hat[:, i])
    else:
        _fit.fit(X_train, y_train)
        y_hat = _fit.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_hat[:, 1])
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
        metrics_dict['AUC'] = roc_auc_score(y_test, y_hat[:, 1])
        metrics_dict['AP'] = average_precision_score(y_test, y_hat[:, 1])
        metrics_dict['Accuracy'] = ((y_test==1) == (y_hat[:, 1] > threshold)).sum() / len(y_test)
        metrics_dict['Null model'] = f1_score(y_test, np.ones(len(y_test)))
        metrics_dict['F1'] = f1_score(y_test, (y_hat[:, 1] > threshold))
        metrics_dict['Balanced Accuracy'] = balanced_accuracy_score(y_test, (y_hat[:, 1] > threshold))
        metrics_dict['Balanced Accuracy Null'] = balanced_accuracy_score(y_test, np.ones(len(y_test)))
    return metrics_dict
