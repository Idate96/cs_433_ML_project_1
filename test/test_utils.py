import unittest
import numpy.testing as npt
import numpy as np
from utils import dataloader, standardize

class TestUtils(unittest.TestCase):

    @unittest.skip
    def test_dataloader(self):
        x, y = dataloader(mode='train')
        self.assertEqual(np.shape(x), (250000, 30))
        self.assertEqual(np.shape(y), (250000,))

    def test_stardardize(self):
        x, y = dataloader(mode='train', reduced=False)
        x = standardize(x)
        npt.assert_array_almost_equal(np.mean(x, axis=0), np.zeros(30))
        npt.assert_array_almost_equal(np.std(x, axis=0), np.ones(30))
