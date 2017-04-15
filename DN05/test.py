from sklearn.metrics import roc_auc_score
from roc.py import AUROC, compute_rho
import random

def do_testing():
    """
    Simple testing function
    """
    trueclass = [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
    probs = [0.95, 0.95, 0.82, 0.82, 0.82, 0.73, 0.61, 0.61, 0.53, 0.38, 0.38, 0.11, 0.11]
    tc = [random.randint(0, 1) for _ in range(20)]
    tp = [random.uniform(0, 1) for _ in range(20)]
    print('   AUROC simple: ', AUROC(trueclass, probs))
    print('ROC AUC sklearn: ', roc_auc_score(trueclass, probs))
    print('   RHO estimate: ', compute_rho(trueclass, probs))
    print('   AUROC random: ', AUROC(tc, tp))
    print('ROC AUC sklearn: ', roc_auc_score(tc, tp))
    print('   RHO estimate: ', compute_rho(tc, tp))

