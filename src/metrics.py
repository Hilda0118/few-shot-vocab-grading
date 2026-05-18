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
    """
    三层评估：
      - Generalized ：全部四类（A1/A2/B1/B2），adapt 之后
      - Base-after   ：仅 A1/A2，adapt 之后（反映是否遗忘）
      - Novel-only   ：仅 B1/B2，adapt 之后（反映 adapt 效果）
    注意：Base-pretrain（fit_base 后、adapt 前）在 run_all.py 里单独计算，
          不经过本函数，直接写入 metrics_rows。
    """
    all_classes   = list(CFG.all_classes)
    base_classes  = list(CFG.base_classes)
    novel_classes = list(CFG.novel_classes)

    # ── Generalized 层 ──
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_f1   = f1_score(y_true, y_pred, average=None,
                        labels=all_classes, zero_division=0)
    try:
        qwk = cohen_kappa_score(label_to_ord(y_true), label_to_ord(y_pred),
                                weights="quadratic")
    except Exception:
        qwk = 0.0

    y_bin    = label_binarize(y_true, classes=all_classes)
    ap_list  = []
    for i in range(len(all_classes)):
        if len(np.unique(y_bin[:, i])) < 2:
            ap_list.append(0.0)
        else:
            ap_list.append(average_precision_score(y_bin[:, i], y_score[:, i]))
    mAP = float(np.mean(ap_list))

    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    # ── Base-after 层（adapt 之后，A1/A2）──
    base_mask = np.isin(y_true, base_classes)
    if base_mask.sum() > 0:
        yb_true  = y_true[base_mask]
        yb_pred  = y_pred[base_mask]
        base_after_f1     = f1_score(yb_true, yb_pred, average="macro",
                                     labels=base_classes, zero_division=0)
        base_after_per_f1 = f1_score(yb_true, yb_pred, average=None,
                                     labels=base_classes, zero_division=0)
        base_after_acc    = accuracy_score(yb_true, yb_pred)
    else:
        base_after_f1     = 0.0
        base_after_per_f1 = [0.0, 0.0]
        base_after_acc    = 0.0

    # ── Novel-only 层（B1/B2）──
    novel_mask = np.isin(y_true, novel_classes)
    if novel_mask.sum() > 0:
        yn_true  = y_true[novel_mask]
        yn_pred  = y_pred[novel_mask]
        novel_macro_f1  = f1_score(yn_true, yn_pred, average="macro",
                                   labels=novel_classes, zero_division=0)
        novel_per_f1    = f1_score(yn_true, yn_pred, average=None,
                                   labels=novel_classes, zero_division=0)
        novel_acc       = accuracy_score(yn_true, yn_pred)
    else:
        novel_macro_f1 = 0.0
        novel_per_f1   = [0.0, 0.0]
        novel_acc      = 0.0

    return {
        # ── Generalized ──
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "qwk":       qwk,
        "mAP":       mAP,
        "f1_A1":     float(per_f1[0]),
        "f1_A2":     float(per_f1[1]),
        "f1_B1":     float(per_f1[2]),
        "f1_B2":     float(per_f1[3]),
        "cm":        cm,
        # ── Base-after（adapt 后，A1/A2）──
        "base_after_f1":     base_after_f1,
        "base_after_acc":    base_after_acc,
        "base_after_f1_A1":  float(base_after_per_f1[0]),
        "base_after_f1_A2":  float(base_after_per_f1[1]),
        # ── Novel-only ──
        "novel_macro_f1": novel_macro_f1,
        "novel_acc":      novel_acc,
        "novel_f1_B1":    float(novel_per_f1[0]),
        "novel_f1_B2":    float(novel_per_f1[1]),
    }
