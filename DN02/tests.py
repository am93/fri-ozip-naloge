import unittest

import numpy as np
from Orange.data import Table
from Orange.preprocess import Normalize

from softmax_reg import SoftmaxLearner, SoftmaxLearner_reg, fit_params


class SoftmaxRegTest(unittest.TestCase):
    def setUp(self):
        self.data = Table("iris")
        self.data = Normalize()(self.data)
        self.X, self.y = self.data.X, self.data.Y
        self.m, self.n = self.X.shape
        self.k = len(self.data.domain.class_var.values)
        self.X1 = np.hstack((np.ones((self.m, 1)), self.X))
        self.theta = np.ones((self.k, self.n + 1)).flatten()
        self.sm = SoftmaxLearner()
        self.sm_reg = SoftmaxLearner_reg()

    # 1. naloga
    def test_cost_reg(self):
        self.assertGreater(self.sm_reg.cost(self.theta, self.X1, self.y), 0)

    def test_grad_reg(self):
        g = self.sm_reg.grad(self.theta, self.X1, self.y)
        self.assertEqual(g.shape, (self.k * (self.n + 1),))

    def test_approx_grad_reg(self):
        g = self.sm_reg.approx_grad(self.theta, self.X1, self.y)
        self.assertEqual(g.shape, (self.k * (self.n + 1),))

    def test_gradient_reg(self):
        np.testing.assert_array_almost_equal(
            self.sm_reg.approx_grad(self.theta, self.X1, self.y),
            self.sm_reg.grad(self.theta, self.X1, self.y))

    def test_fit_reg(self):
        model = self.sm_reg.fit(self.X, self.y)
        self.assertEqual(model.theta.shape, (self.k, self.n + 1))

    def test_probability_reg(self):
        model = self.sm_reg.fit(self.X, self.y)
        y_prob = model.predict(self.X)
        np.testing.assert_array_almost_equal(np.sum(y_prob, axis=1),
                                             np.ones(len(self.X)))

    def test_predict_reg(self):
        self.sm_reg.lambda_ = 0.01
        model_1 = self.sm_reg.fit(self.X[::2], self.y[::2])
        y_prob_1 = model_1.predict(self.X[1::2])
        y_pred_1 = np.argmax(y_prob_1, axis=1)
        ca_1 = sum(1 for y_real, y_p in zip(self.y[1::2], y_pred_1)
                   if y_real == y_p) / len(y_pred_1)
        self.sm_reg.lambda_ = 1.
        model_2 = self.sm_reg.fit(self.X[::2], self.y[::2])
        y_prob_2 = model_2.predict(self.X[1::2])
        y_pred_2 = np.argmax(y_prob_2, axis=1)
        ca_2 = sum(1 for y_real, y_p in zip(self.y[1::2], y_pred_2)
                   if y_real == y_p) / len(y_pred_2)
        self.assertGreater(ca_1, ca_2)
        self.assertGreater(ca_1, 0.95)

    def test_cost_compare(self):
        sm = SoftmaxLearner()
        sm_reg = SoftmaxLearner_reg(lambda_=1.)
        self.assertLess(sm.cost(self.theta, self.X1, self.y),
                        sm_reg.cost(self.theta, self.X1, self.y))
        sm_reg.lambda_ = 0
        self.assertEqual(sm.cost(self.theta, self.X1, self.y),
                         sm_reg.cost(self.theta, self.X1, self.y))

    def test_grad_compare(self):
        sm_reg = SoftmaxLearner_reg(lambda_=0)
        np.testing.assert_array_almost_equal(
            sm_reg.grad(self.theta, self.X1, self.y),
            self.sm.grad(self.theta, self.X1, self.y))

    def test_weights_smaller(self):
        sm1 = SoftmaxLearner_reg(lambda_=1)
        sm10 = SoftmaxLearner_reg(lambda_=10)
        weights1 = sm1(self.data).theta.flatten()
        weights10 = sm10(self.data).theta.flatten()
        self.assertLess(weights10.dot(weights10), weights1.dot(weights1))

    # 2. naloga
    def test_fit_params(self):
        best_lambda = fit_params(SoftmaxLearner_reg, 'lambda_',
                                 [0.0001, 1, 10000], self.data)
        self.assertEqual(best_lambda, 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
