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

    def mysigma(self, x):
        """
        My softmax function. Always check that you provide correctly oriented data (ignore - solved with slicing).
        I subtracted max value to prevent overflow at calculation of exponent - it may cause undeflow, but that is
        not a problem.
        """
        tmpx = np.exp(x - np.max(x, axis=1)[:, None])
        return tmpx / np.sum(tmpx,axis=1)[:, None]

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
        num_classes = len(np.unique(y)) # predpostavljamo da so vsi razredi prisotni
        X = np.column_stack((np.ones(X.shape[0]), X))
        theta = np.ones(num_classes * X.shape[1]) * 1e-9;
        result = fmin_l_bfgs_b(self.cost, theta, self.grad, args=(X,y))[0]
        return SoftmaxModel(result.reshape((-1, X.shape[1])));


class SoftmaxModel(Model):
    def __init__(self, theta):
        self.theta = theta

    def mysigma(self, x):
        """
        My softmax function. Always check that you provide correctly oriented data (ignore - solved with slicing).
        I subtracted max value to prevent overflow at calculation of exponent - it may cause undeflow, but that is
        not a problem.
        """
        tmpx = np.exp(x - np.max(x, axis=1)[:, None])
        return tmpx / np.sum(tmpx,axis=1)[:, None]

    def predict(self, X):
        """
        Args:
            X (np.ndarray): data of shape [n_examples, n_features]

        Returns:
            np.ndarray: Predictions of shape [n_examples, n_classes]
        """
        X = np.column_stack((np.ones(X.shape[0]), X))
        return self.mysigma(X.dot(self.theta.T))


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
        # samo dodam vrstico nicel na zacetek (to je prva vrstica, ki se je ne ucimo)
        theta = np.concatenate((np.zeros(X.shape[1]), theta))
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
        # samo dodam vrstico nicel na zacetek (to je prva vrstica, ki se je ne ucimo)
        theta = np.concatenate((np.zeros(X.shape[1]), theta))
        # vrniti moram gradient samo za tiste thete, ki sem jih dobil - brez prve vrstice
        return super().grad(theta, X, y)[X.shape[1]:]

    def fit(self, X, y, W=None):
        """
        Args:
            X (np.ndarray): data of shape [n_examples, n_features]
            y (np.ndarray): target variable of shape [n_examples]
            W (np.ndarray): Orange weights - ignore for this exercise

        Returns:
            SoftmaxModel: Orange's classification model
        """
        num_classes = len(np.unique(y)) - 1  # zmanjsamo stevilo razredov za 1
        X = np.column_stack((np.ones(X.shape[0]), X))
        theta = np.ones(num_classes * X.shape[1]) * 1e-9;
        result = fmin_l_bfgs_b(self.cost, theta, self.grad, args=(X, y))[0]
        result = np.concatenate((np.zeros(X.shape[1]), result)) # dodamo nicle za prvo vrstico
        return SoftmaxModel(result.reshape((-1, X.shape[1])));


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
