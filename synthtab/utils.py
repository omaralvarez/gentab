import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_accuracy(y_true, y_pred):
    return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))


def compute_f1_p_r(y_true, y_pred, average):
    return precision_recall_fscore_support(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average=average
    )
