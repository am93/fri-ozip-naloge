import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import random


def compute_roc_curve(true_class, prob, target=1):

    # sort examples by descending probability
    examples = sorted(zip(true_class, prob), key=lambda x: x[1], reverse=True)
    tpr, fpr = [0], [0]

    n_pos = sum(true_class)
    n_neg = len(true_class) - n_pos

    crr_prob = examples[0][1]
    tp = 0
    fp = 0

    # loop through examples
    for ex in examples:
        if ex[1] != crr_prob:
            tpr.append(tpr[-1] + tp / n_pos)
            fpr.append(fpr[-1] + fp / n_neg)
            tp, fp = 0, 0
            crr_prob = ex[1]

        if ex[1] == crr_prob and ex[0] == target:
            tp += 1
        elif ex[1] == crr_prob and ex[0] != target:
            fp += 1

    # last points
    tpr.append(1)
    fpr.append(1)

    return tpr, fpr


def AUROC(real, class_prob):
    """
    Function computes area under ROC curve.
    """
    examples = sorted(zip(real, class_prob), key=lambda x: x[1], reverse=True)
    n_pos = 0
    n_neg = 0
    v_auc = 0
    last_zero_prob = -1
    for ex in examples:
        if ex[0] == 1:
            n_pos += 1
            if ex[1] == last_zero_prob:
                v_auc += 0.5
        if ex[0] == 0:
            n_neg += 1
            last_zero_prob = ex[1]
            v_auc += n_pos

    return v_auc / (n_pos * n_neg)




if __name__ == '__main__':
    trueclass = [1,1,0,1,1,1,0,0,1,0,1,0,0]
    probs = [0.95, 0.95, 0.82, 0.82, 0.82, 0.73, 0.61, 0.61, 0.53, 0.38, 0.38, 0.11, 0.11]
    tc = [random.randint(0,1) for _ in range(20)]
    tp = [random.uniform(0, 1) for _ in range(20)]
    print(AUROC(trueclass, probs))
    print(roc_auc_score(trueclass, probs))
    tpr, fpr = compute_roc_curve(trueclass, probs)
    plt.plot(fpr, tpr)
    plt.show()

    print(AUROC(tc, tp))
    print(roc_auc_score(tc, tp))

    y_real = [random.randint(0, 1) for i in range(20)]
    y_pred_my = [[0, random.uniform(0, 1)] for _ in range(20)]
    y_pred_builtin = [x[1] for x in y_pred_my]
    y_pred_my = [[1 - x[1], x[1]] for x in y_pred_my]
    print(float(roc_auc_score(y_real, y_pred_builtin)), AUC(y_real, y_pred_my))