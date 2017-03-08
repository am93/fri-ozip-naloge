import numpy as np
import matplotlib.pyplot as plt
import Orange
from math import ceil

from softmax import SoftmaxLearner


class SoftmaxLearner_reg(SoftmaxLearner):
    def __init__(self, lambda_=1.0, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.lambda_ = lambda_

    def cost(self, theta, X, y):
        """
        Call parent implementation of cost function and add regularization part. Because result returned by parent
        is already multiplied with -1, we add regularization part and not subtract it !
        """
        reg_part = 0.5 * self.lambda_ * np.sum(theta ** 2)
        return super().cost(theta, X, y) + reg_part

    def grad(self, theta, X, y):
        """
        Similar as previous function - call parent implementation and add regularization part.
        """
        reg_part = self.lambda_ * theta
        return super().grad(theta, X, y) + reg_part


def fit_params(learner, param_name, values, data):
    """
    Find best parameter for learner on data.
    Use cross-validation with k=3 and AUC for scoring.

    Args:
        learner (Orange.classification.Learner): Learner
        param_name (str): the name of the parameter we are fitting
        values (list): list of parameter values to consider
        data (Orange.data.Table): data table used for evaluation

    Returns:
        float: The best parameter value
    """
    results = []
    for value in values:
        res = Orange.evaluation.CrossValidation(data, [learner(**{param_name: value})], k=3)
        auc = Orange.evaluation.scoring.AUC(res)
        results.append((value, auc[0]))

    results = sorted(results, key = lambda x: x[1], reverse=True)
    return results[0][0]


def plot_mnist_weights(theta, filename='fig.pdf'):
    """
    Plot weights for all classes in a single image with subplots.
    For each class plot a 28x28 image.

    Args:
        theta (np.ndarray): model parameters of shape [n_classes * n_features]
        filename (str): save image to filename
    """
    fig, axes = plt.subplots(nrows=ceil(theta.shape[0]/3), ncols=3, subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        if i >= theta.shape[0]:
            ax.axis('off')
            continue
        tmp_theta = theta[i][1:].reshape((-1, 28))
        im = ax.imshow(tmp_theta, interpolation=None, cmap='seismic')
        fig.colorbar(im, ax=ax)
        ax.set_title('class = '+str(i))

    fig.tight_layout(w_pad=1)
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    data = Orange.data.Table('iris')

    # Regularization
    # weights.dot(weights)  is the same as  sum(w**2 for w in weights)
    sm = SoftmaxLearner_reg(lambda_=0)
    weights = sm(data).theta.flatten()
    print('No regularization:', weights.dot(weights))

    sm = SoftmaxLearner_reg(lambda_=1)
    weights = sm(data).theta.flatten()
    print('Weaker regularization:', weights.dot(weights))

    sm = SoftmaxLearner_reg(lambda_=10)
    weights = sm(data).theta.flatten()
    print('Stronger regularization:', weights.dot(weights))

    # Parameter optimization (try also on mnist data)
    best_lambda = fit_params(SoftmaxLearner_reg, 'lambda_', [0.1, 1, 10, 100], data)
    print('Best lambda_:', best_lambda)

    # Plot weights for MNIST data
    data = Orange.data.Table('mnist-1k.tab')
    sm = SoftmaxLearner_reg(lambda_=1)
    m = sm(data)
    plot_mnist_weights(m.theta, 'lambda_1.pdf')
    sm = SoftmaxLearner_reg(lambda_=1e7)
    m = sm(data)
    plot_mnist_weights(m.theta, 'lambda_1e7.pdf')
