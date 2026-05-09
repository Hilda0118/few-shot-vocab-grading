import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score,
    cohen_kappa_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from .config import CFG


def label_to_ord(y):
    m = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}
    return np.array([m[i] for i in y])


def eval_classification(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_f1 = f1_score(y_true, y_pred, average=None, labels=list(CFG.all_classes))
    qwk = cohen_kappa_score(label_to_ord(y_true), label_to_ord(y_pred), weights="quadratic")

    y_bin = label_binarize(y_true, classes=list(CFG.all_classes))
    ap_list = []
    for i in range(len(CFG.all_classes)):
        # 若该类在y_true中全0或全1，AP可能不稳定，这里兜底
        if len(np.unique(y_bin[:, i])) < 2:
            ap_list.append(0.0)
        else:
            ap_list.append(average_precision_score(y_bin[:, i], y_score[:, i]))
    mAP = float(np.mean(ap_list))

    cm = confusion_matrix(y_true, y_pred, labels=list(CFG.all_classes))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "qwk": qwk,
        "mAP": mAP,
        "f1_A1": per_f1[0],
        "f1_A2": per_f1[1],
        "f1_B1": per_f1[2],
        "f1_B2": per_f1[3],
        "cm": cm
    }
