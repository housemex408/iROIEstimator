import unittest
import os
import sys
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
from iROIEstimator import iROIEstimator

class TestIROIEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = iROIEstimator('angular/angular.js')
        self.estimator.amount_invested = 182389
        self.estimator.amount_returned = 1863864
        self.estimator.investment_gain = 1681475
        self.estimator.roi = 9.2

    def test_calculate_investment_gain(self):
        gain = self.estimator.calculate_investment_gain()
        self.assertEqual(gain, 1681475)

    def test_calculate_ROI(self):
        roi = self.estimator.calculate_ROI()
        self.assertEqual(roi, 9.22)

    def test_calculate_annualized_ROI(self):
        ann_roi = self.estimator.calculate_annualized_ROI()
        self.assertEqual(ann_roi, 1.17)

if __name__ == '__main__':
    unittest.main()
