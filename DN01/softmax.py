import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from Orange.classification import Learner, Model
from Orange.preprocess import (RemoveNaNClasses, Normalize, Continuize,
Impute, RemoveNaNColumns)


class SoftmaxLearner(Learner):
    """
    Implementation of softmax regression with k*(n+1) parameters
    trained using L-BFGS optimization.
    """
    name = 'softmax'
    preprocessors = [RemoveNaNClasses(),
                     Normalize(),
                     Continuize(),
                     Impute(),
                     RemoveNaNColumns()]

    def __init__(self, preprocessors=None):
        super().__init__(preprocessors=preprocessors)

    def mysigma(self, z):
        """
        My softmax function. Always check that you provide correctly oriented data.
        """
        return np.exp(z) / np.sum(np.exp(z),axis=1)[:, None]

    def cost(self, theta, X, y):
        """
        Args:
            theta (np.ndarray): model parameters of shape [n_classes * n_features]
            X (np.ndarray): data of shape [n_examples, n_features]
            y (np.ndarray): target variable of shape [n_examples]

        Returns:
            float: The value of cost function evaluated with given parameters.
        """
        #################################################################################################
        # Theta pretvorim iz dolgega vektorja v matricno obliko, nato pripravim indikatorsko funkcijo
        #################################################################################################
        theta = theta.reshape((-1, X.shape[1]))
        indicator = np.identity(theta.shape[0])[y.astype(int)]
        return -(np.sum(indicator * np.log(self.mysigma(X.dot(theta.T)))))

    def grad(self, theta, X, y):
        """
        Args:
            theta (np.ndarray): model parameters of shape [n_classes * n_features]
            X (np.ndarray): data of shape [n_examples, n_features]
            y (np.ndarray): target variable of shape [n_examples]

        Returns:
            np.ndarray: Gradients wrt. all model's parameters of shape
                [n_classes * n_features]
        """
        theta = theta.reshape((-1, X.shape[1]))
        indicator = np.identity(theta.shape[0])[y.astype(int)]
        return -(X.T.dot((indicator - self.mysigma(X.dot(theta.T))))).T.flatten()

    def approx_grad(self, theta, X, y, eps=1e-5):
        """
        Args:
            theta (np.ndarray): model parameters of shape [n_classes * n_features]
            X (np.ndarray): data of shape [n_examples, n_features]
            y (np.ndarray): target variable of shape [n_examples]
            eps (float): value offset for gradient estimation

        Returns:
            np.ndarray: Gradients wrt. all model's parameters of shape
                [n_classes * n_features]
        """
        result = []
        for i in range(len(theta)):
            crr = np.zeros(len(theta))
            crr[i] = 1
            result.append((self.cost(theta + (crr * eps),X,y) - self.cost(theta - (crr * eps),X,y)) / (2 * eps))

        return np.array(result)

    def fit(self, X, y, W=None):
        """
        Args:
            X (np.ndarray): data of shape [n_examples, n_features]
            y (np.ndarray): target variable of shape [n_examples]
            W (np.ndarray): Orange weights - ignore for this exercise

        Returns:
            SoftmaxModel: Orange's classification model
        """
        #######################################################################
        # TODO: implement this function
        #######################################################################
        pass


class SoftmaxModel(Model):
    def __init__(self, theta):
        self.theta = theta

    def predict(self, X):
        """
        Args:
            X (np.ndarray): data of shape [n_examples, n_features]

        Returns:
            np.ndarray: Predictions of shape [n_examples, n_classes]
        """
        #######################################################################
        # TODO: implement this function
        #######################################################################
        pass


class SoftmaxLearner_p(SoftmaxLearner):
    """
    Implementation with (k-1)*(n+1) parameters.
    """

    def cost(self, theta, X, y):
        """
        Args:
            theta (np.ndarray): model parameters of shape [(n_classes - 1) * n_features]
            X (np.ndarray): data of shape [n_examples, n_features]
            y (np.ndarray): target variable of shape [n_examples]

        Returns:
            float: The value of cost function evaluated with given parameters.
        """
        #######################################################################
        # TODO: implement this function
        #######################################################################
        return super().cost(theta, X, y)

    def grad(self, theta, X, y):
        """
        Args:
            theta (np.ndarray): model parameters of shape [(n_classes - 1) * n_features]
            X (np.ndarray): data of shape [n_examples, n_features]
            y (np.ndarray): target variable of shape [n_examples]

        Returns:
            np.ndarray: Gradients wrt. all model's parameters of shape
                [(n_classes - 1) * n_features]
        """
        #######################################################################
        # TODO: implement this function
        #######################################################################
        return super().grad(theta, X, y)

    def fit(self, X, y, W=None):
        """
        Args:
            X (np.ndarray): data of shape [n_examples, n_features]
            y (np.ndarray): target variable of shape [n_examples]
            W (np.ndarray): Orange weights - ignore for this exercise

        Returns:
            SoftmaxModel: Orange's classification model
        """
        #######################################################################
        # TODO: implement this function
        #######################################################################
        return super().fit(X, y, W)


if __name__ == '__main__':
    import Orange
    data = Orange.data.Table('iris')
    sm = SoftmaxLearner()
    m = sm(data[::2])
    print(m.theta)
    print(m(data[1::2]))

    # _p
    sm = SoftmaxLearner_p()
    m = sm(data[::2])
    print(m.theta)
    print(m(data[1::2]))
