import unittest

import numpy as np
from Orange.data import Table
from Orange.preprocess import Normalize

from softmax import SoftmaxLearner, SoftmaxLearner_p


class SoftmaxTest(unittest.TestCase):
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

    # 1. naloga
    def test_cost(self):
        self.assertGreater(self.sm.cost(self.theta, self.X1, self.y), 0)

    def test_grad(self):
        g = self.sm.grad(self.theta, self.X1, self.y)
        self.assertEqual(g.shape, (self.k * (self.n + 1),))

    def test_approx_grad(self):
        g = self.sm.approx_grad(self.theta, self.X1, self.y)
        self.assertEqual(g.shape, (self.k * (self.n + 1),))

    def test_gradient(self):
        np.testing.assert_array_almost_equal(
            self.sm.approx_grad(self.theta, self.X1, self.y),
            self.sm.grad(self.theta, self.X1, self.y))

    # 2. naloga
    def test_fit(self):
        model = self.sm.fit(self.X, self.y)
        self.assertEqual(model.theta.shape, (self.k, self.n + 1))

    def test_fit_cost(self):
        model = self.sm.fit(self.X, self.y)
        cost1 = self.sm.cost(self.theta, self.X1, self.y)
        cost_fitted = self.sm.cost(model.theta.flatten(), self.X1, self.y)
        self.assertGreater(cost1, cost_fitted)

    # 3. naloga
    def test_predict(self):
        model = self.sm.fit(self.X, self.y)
        y_prob = model.predict(self.X)
        self.assertEqual(y_prob.shape, (self.m, self.k))

    def test_probability(self):
        model = self.sm.fit(self.X, self.y)
        y_prob = model.predict(self.X)
        self.assertTrue((y_prob >= 0).all())
        self.assertTrue((y_prob <= 1).all())
        np.testing.assert_array_almost_equal(np.sum(y_prob, axis=1),
                                             np.ones(len(self.X)))

    def test_accuracy(self):
        model = self.sm.fit(self.X, self.y)
        y_prob = model.predict(self.X)
        y_pred = np.argmax(y_prob, axis=1)
        self.assertGreater(sum(1 for y_real, y_p in zip(self.y, y_pred)
                               if y_real == y_p) / self.m, 0.95)

    # 4. naloga
    def test_cost_p(self):
        self.assertGreater(self.sm_p.cost(self.theta_p, self.X1, self.y), 0)

    def test_grad_p(self):
        g = self.sm_p.grad(self.theta_p, self.X1, self.y)
        self.assertEqual(g.shape, ((self.k - 1) * (self.n + 1),))

    def test_approx_grad_p(self):
        g = self.sm_p.approx_grad(self.theta_p, self.X1, self.y)
        self.assertEqual(g.shape, ((self.k - 1) * (self.n + 1),))

    def test_gradient_p(self):
        np.testing.assert_array_almost_equal(
            self.sm_p.approx_grad(self.theta_p, self.X1, self.y),
            self.sm_p.grad(self.theta_p, self.X1, self.y))

    def test_fit_p(self):
        model = self.sm_p.fit(self.X, self.y)
        self.assertEqual(model.theta.shape, (self.k, self.n + 1))

    def test_fit_cost_p(self):
        model = self.sm_p.fit(self.X, self.y)
        cost1 = self.sm_p.cost(self.theta_p, self.X1, self.y)
        cost_fitted = self.sm_p.cost(model.theta[1:].flatten(), self.X1, self.y)
        self.assertGreater(cost1, cost_fitted)

    def test_fit_p_compare(self):
        theta = self.sm.fit(self.X, self.y).theta
        theta_p = self.sm_p.fit(self.X, self.y).theta
        self.assertGreater(np.sum(np.abs(theta - theta_p)), 0)
        self.assertAlmostEqual(self.sm.cost(theta, self.X1, self.y),
                               self.sm.cost(theta_p, self.X1, self.y), 4)

    def test_fit_p_check_0(self):
        model = self.sm_p.fit(self.X, self.y)
        self.assertEqual(sum(model.theta[0]), 0)
        model = self.sm.fit(self.X, self.y)
        self.assertNotEqual(sum(model.theta[0]), 0)

    def test_predict_p(self):
        model = self.sm_p.fit(self.X, self.y)
        y_prob = model.predict(self.X)
        self.assertEqual(y_prob.shape, (self.m, self.k))

    def test_probability_p(self):
        model = self.sm_p.fit(self.X, self.y)
        y_prob = model.predict(self.X)
        self.assertTrue((y_prob >= 0).all())
        self.assertTrue((y_prob <= 1).all())
        np.testing.assert_array_almost_equal(np.sum(y_prob, axis=1),
                                             np.ones(len(self.X)))

    def test_accuracy_p(self):
        model = self.sm_p.fit(self.X, self.y)
        y_prob = model.predict(self.X)
        y_pred = np.argmax(y_prob, axis=1)
        self.assertGreater(sum(1 for y_real, y_p in zip(self.y, y_pred)
                               if y_real == y_p) / self.m, 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
