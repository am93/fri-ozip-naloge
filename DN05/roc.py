import matplotlib.pyplot as plt
import random
import csv


def compute_roc_curve(true_class, prob, target=1):
    """
    Function computes ROC curve points and returns them in two arrays - true positive rate and false positive rate.
    :param true_class: array of class variables
    :param prob: prediciton of target class (1) probability
    :param target: target class value
    :return: tpr, fpr - two arrays
    """

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
    examples = sorted(zip(real, class_prob), key=lambda x: (-x[1],-x[0]))
    n_pos = 0
    n_neg = 0
    v_auc = 0
    crr_pos, last_pos = 0, -1
    for ex in examples:
        if ex[0] == 1:
            n_pos += 1
            if last_pos == ex[1]:
                crr_pos += 1
            else:
                last_pos = ex[1]
                crr_pos = 1
        if ex[0] == 0:
            if last_pos == ex[1]:
                v_auc -= 0.5 * crr_pos
            n_neg += 1
            v_auc += n_pos

    return v_auc / (n_pos * n_neg)


def compute_rho(real, class_prob, num_iter=1000):
    """
    Compute estimation RHO of AUROC by random sampling example pairs for num_iters
    :param real: list of class variables
    :param class_prob: prediction probabilities
    :param num_iter: number of iterations
    :return: rho estimate
    """
    examples = list(zip(real, class_prob))
    pos_ex = [ex[1] for ex in examples if ex[0] == 1]
    neg_ex = [ex[1] for ex in examples if ex[0] == 0]

    iter = 0
    rho = 0
    while iter < num_iter:
        idx1 = random.randint(0, len(pos_ex) - 1)
        idx2 = random.randint(0, len(neg_ex) - 1)
        if pos_ex[idx1] > neg_ex[idx2]:
            rho += 1
        elif pos_ex[idx1] == neg_ex[idx2]:
            rho += 0.5
        iter += 1

    return rho/num_iter


def read_input_csv(filename):
    """
    Function reads input csv file and returns array of class variables and prediction probabilities
    :param filename: input CSV filename
    :return: y, prob
    """
    y, prob = [], []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'id':
                continue
            y.append(int(row[1]))
            prob.append(float(row[2]))

    return y, prob


def plot_results(tpr, fpr, auc, rho):
    """
    Plot results
    :param tpr: true positive rate list
    :param fpr: false positive rate list
    :param auc: compute AUC
    :param rho: estimated AUC
    """
    plt.plot(fpr, tpr)
    plt.title("Krivulja ROC, AUC=%.4f, r=%.4f" % (auc, rho))
    plt.show()


if __name__ == '__main__':
    y, prob = read_input_csv('roc-podatki.csv')
    auc = AUROC(y, prob)
    rho = compute_rho(y, prob)
    tpr, fpr = compute_roc_curve(y, prob)
    plot_results(tpr, fpr, auc, rho)