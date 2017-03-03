import numpy as np
from Orange.data import Table
from Orange.preprocess import Normalize

from softmax import SoftmaxLearner, SoftmaxLearner_p

class SoftmaxTest():
    def setUp(self):
        data = Table("iris")
        data = Normalize()(data)
        self.X, self.y = data.X, data.Y
        self.m, self.n = self.X.shape
        self.k = len(data.domain.class_var.values)
        self.X1 = np.hstack((np.ones((self.m, 1)), self.X))
        self.theta = np.ones((self.k, self.n + 1)).flatten()
        self.theta_p = np.ones((self.k - 1, self.n + 1)).flatten()
        self.sm = SoftmaxLearner()
        self.sm_p = SoftmaxLearner_p()

derp = SoftmaxTest()
derp.setUp()
print(derp.sm.cost(derp.theta, derp.X1, derp.y))
print(derp.sm.grad(derp.theta, derp.X1, derp.y))
print(derp.sm.approx_grad(derp.theta, derp.X1, derp.y))
print(derp.sm.fit(derp.X, derp.y).theta)