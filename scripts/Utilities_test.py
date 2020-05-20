import unittest
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
import numpy as np

class TestStringMethods(unittest.TestCase):
    def test_format_PRED(self):
        fmtString = utils.format_PRED("25", .25)
        self.assertEqual(fmtString, "Pred - PRED (25): 25.00%")

    def test_format_perf_metric(self):
        label = "Model - R Squared"
        fmtString = utils.format_perf_metric(label, .25)
        self.assertEqual(fmtString, "Model - R Squared: 0.25")

    def test_isRegularVersion_returnTrue(self):
        series = {
          c.VERSION: ["v3.4.5.6", "v3.4.5-rc1", "v3.4"],
          c.T_MODULE: [9876, 6745, 3659]
          }
        df = pd.DataFrame(series)
        filtered_versions = utils.isRegularVersion(df)

        self.assertTrue(len(filtered_versions) == 2)

    def test_is_all_same(self):
        data = [1,1,1,1,1,1]
        df = pd.Series(data)

        result = utils.is_all_same(df)
        self.assertTrue(result)

    def test_percentage_nan(self):
        series = {
          c.VERSION: ["v3.4.5.6", "v3.4.5-rc1", "v3.4"],
          c.T_MODULE: [9876, 9876, np.nan]
          }
        df = pd.DataFrame(series)

        result = utils.percentage_nan(df)

        self.assertEqual(result, 0.167)

if __name__ == '__main__':
    unittest.main()
