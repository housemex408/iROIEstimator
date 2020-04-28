import unittest
import os
import sys
sys.path.append(os.path.abspath("/Users/alvaradojo/Documents/Github/iROIEstimator/scripts"))
import Utilities as utils

class TestStringMethods(unittest.TestCase):
    def test_format_PRED(self):
        value = .25
        percentage = 25

        fmtString = utils.format_PRED("25", .25)
        self.assertEqual(fmtString, "Pred - PRED (25): 25.00%")

    def test_format_perf_metric(self):
        value = .25
        label = "Model - R Squared"

        fmtString = utils.format_perf_metric(label, .25)
        self.assertEqual(fmtString, "Model - R Squared: 0.25")

if __name__ == '__main__':
    unittest.main()