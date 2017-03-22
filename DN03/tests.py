import unittest

import numpy as np
from Orange.data import Table

from pca import *


class PCATest(unittest.TestCase):
    def setUp(self):
        self.data = Table("iris")

    def test_pca_full(self):
        vecs = pca_full(self.data.X)
        self.assertEqual(vecs.shape, (4, 4))
        np.testing.assert_array_almost_equal(vecs.T.dot(vecs), np.eye(4))

        projected = project_data(self.data.X, vecs)
        variance = np.var(projected, axis=0)
        self.assertGreater(variance[0], variance[-1])

    def test_pca_2d(self):
        vecs = pca_2d(self.data.X)
        self.assertEqual(vecs.shape, (4, 2))
        np.testing.assert_array_almost_equal(vecs.T.dot(vecs), np.eye(2))

    def test_pca_compare(self):
        vecs_full = pca_full(self.data.X)[:, :2]
        vecs_2d = pca_2d(self.data.X)
        np.testing.assert_array_almost_equal(np.abs(vecs_full.T.dot(vecs_2d)),
                                             np.eye(2))

    def test_gram_schmidt_orthogonalize(self):
        vecs = np.random.random((100, 2)) - 0.5
        vecs = gram_schmidt_orthogonalize(vecs)
        np.testing.assert_array_almost_equal(vecs.T.dot(vecs), np.eye(2))

if __name__ == "__main__":
    unittest.main(verbosity=2)
