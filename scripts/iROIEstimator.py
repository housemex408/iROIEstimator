import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from fbprophet import Prophet
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
from Effort import Effort


class iROIEstimator:
    # input = "../../exports"
    input = "scripts/exports"
    output = "scripts/notebook/results"
    results_file = "{directory}/prediction_results.csv".format(directory=output)
    performance_measures_file = "{directory}/performance_measures.csv".format(directory=output)
    roi_measures_file = "{directory}/roi_measures.csv".format(directory=output)
    TASK_LIST = c.TASK_LIST
    # TASK_LIST = ["BUG"]
    results_header = [c.DATE, c.PROJECT, c.MODEL, c.TASK, c.NT, c.NO, c.T_MODULE, c.OBSERVED, c.PREDICTED, c.DIFFERENCE, c.PERCENT_ERROR]
    performance_measures_header = [c.PROJECT, c.MODEL, c.TASK, c.R_SQUARED, c.R_SQUARED_ADJ, c.MAE, c.MSE, c.RMSE, c.PRED_25, c.PRED_50, c.T_RECORDS]
    roi_header = [c.PROJECT, c.MODEL, c.AMOUNT_INVESTED, c.AMOUNT_RETURNED, c.ROI, c.ANNUALIZED_ROI]
    results = pd.DataFrame(columns = results_header)
    performance_measures = pd.DataFrame(columns = performance_measures_header)
    results.to_csv(results_file, index=False)
    performance_measures.to_csv(performance_measures_file, index=False)
    roi_measures = pd.DataFrame(columns = roi_header)
    roi_measures.to_csv(roi_measures_file, index=False)

    def __init__(self, project, prediction_years=3):
        self.project_name = project.split('/')[1]
        self.file_template = "{cwd}/{project_name}/{project_name}_dataset_{task}.csv"
        self.line_cc = None
        self.line_ec = None
        self.module_cc = None
        self.module_ec = None
        self.prediction_years = prediction_years
        self.predicton_months = self.prediction_years * 12
        self.task_forecasted_effort = {};
        self.amount_invested = 0
        self.amount_returned = 0
        self.investment_gain = 0
        self.roi = 0
        self.annualized_roi = 0

    def execute(self):
        for task in self.TASK_LIST:
            tasks = self.file_template.format(cwd=self.input, project_name=self.project_name, task = task)
            df = pd.read_csv(tasks)

            df = df.dropna(subset=[c.T_MODULE])
            df = df.dropna(subset=[c.DATE])
            df = df.dropna(subset=[c.TASK])
            df[c.DATE] = pd.to_datetime(df[c.DATE])
            df = df.set_index(c.DATE)
            df.index.name = c.DATE
            df.index = df.index.strftime('%Y-%m-%d')
            if df.isna().values.any():
                df.fillna(0, inplace=True)

            t_records = len(df)

            # Edge case when < 2 tasks detected
            if t_records < 2:
                break

            results, performance_measures = self.predict_effort(task, df)

            self.results = pd.concat([self.results, results])
            self.performance_measures = pd.concat([self.performance_measures, performance_measures])

            self.forecast_effort(df, task)
        self.display_forecast(self.prediction_years)
        self.save_results_performance_measures()
        self.calculate_results()
        self.save_results_roi_measures()

    def predict_effort(self, task, df):
        # LINE_CC
        # self.line_cc = Effort(self.project_name, c.LINE_CC, task, df)
        # line_cc_results = self.line_cc.predict_effort()
        # print("\n{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, c.LINE_CC, line_cc_results.size))
        # self.line_cc.calculate_perf_measurements()
        # line_cc_output = self.line_cc.create_output_df()
        # print(line_cc_output.head())

        # LINE_EC
        # self.line_ec = Effort(self.project_name, c.LINE_EC, task, df)
        # line_ec_results = self.line_ec.predict_effort()
        # print("\n{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, c.LINE_EC, line_ec_results.size))
        # self.line_ec.calculate_perf_measurements()
        # line_ec_output = self.line_ec.create_output_df()
        # print(line_ec_output.head())

        # MODULE_CC
        self.module_cc = Effort(self.project_name, c.MODULE_CC, task, df)
        module_cc_results = self.module_cc.predict_effort()
        self.module_cc.calculate_perf_measurements()
        module_cc_output = self.module_cc.create_output_df()

        # MODULE_EC
        self.module_ec = Effort(self.project_name, c.MODULE_EC, task, df)
        module_ec_results = self.module_ec.predict_effort()
        self.module_ec.calculate_perf_measurements()
        module_ec_output = self.module_ec.create_output_df()

        results_df = pd.concat([module_cc_results, module_ec_results])
        output_df = pd.concat([module_cc_output, module_ec_output])

        return results_df, output_df

    def forecast_effort(self, df, task):
        self.module_cc.forecast_module_effort(self.predicton_months)
        self.module_ec.forecast_module_effort(self.predicton_months)

        self.task_forecasted_effort[task] = {
            c.MODULE_CC: self.module_cc.calculate_total_effort(self.prediction_years),
            c.MODULE_EC: self.module_ec.calculate_total_effort(self.prediction_years)
        }

    def display_forecast(self, predictionYears):
        for key in self.task_forecasted_effort:
            print("\n{0} - {1} Forecasted Effort: \n".format(self.project_name, key))
            print(self.task_forecasted_effort[key][c.MODULE_CC].T)
            print(self.task_forecasted_effort[key][c.MODULE_EC].T)

            self.results[c.DATE] = self.results.index
            self.results.sort_values(by=[c.DATE, c.PROJECT, c.MODEL, c.TASK], ascending = True, inplace=True)
            self.performance_measures.sort_values(by=[c.PROJECT, c.MODEL, c.TASK], inplace=True)
            print(self.results)
            print(self.performance_measures)

    def save_results_performance_measures(self):
        self.results.to_csv(self.results_file, header=False, mode = 'a', index=False)
        self.performance_measures.to_csv(self.performance_measures_file, header=False, mode = 'a', index=False)

    def calculate_investment_gain(self):
        self.investment_gain = round(self.amount_returned - self.amount_invested, 2)
        return self.investment_gain

    def calculate_ROI(self):
        self.roi = round(((self.investment_gain + self.amount_invested) / self.amount_invested) - 1, 2)
        return self.roi

    def calculate_annualized_ROI(self):
        self.annualized_roi = round(pow(1 + self.roi, 1 / self.prediction_years) - 1, 2)
        return self.annualized_roi

    def calculate_results(self):
        effort_cc = 0.0
        effort_ec = 0.0
        for key in self.task_forecasted_effort:
            effort_cc = self.task_forecasted_effort[key][c.MODULE_CC].values.sum()
            effort_ec = self.task_forecasted_effort[key][c.MODULE_EC].values.sum()
            self.amount_invested = self.amount_invested + effort_cc
            self.amount_returned = self.amount_returned + effort_ec
            print("{0} - {1} CC Forecasted Effort: {2}".format(self.project_name, key, round(effort_cc), 2))
            print("{0} - {1} EC Forecasted Effort: {2}".format(self.project_name, key, round(effort_ec), 2))

        print("{0} - Core Contributor Forecasted Effort Over {1} years: {2}".format(self.project_name, self.prediction_years, round(self.amount_invested, 2)))
        print("{0} - External Contributor Forecasted Effort Over {1} years: {2}".format(self.project_name, self.prediction_years, round(self.amount_returned, 2)))

        self.calculate_investment_gain()
        self.calculate_ROI()
        self.calculate_annualized_ROI()

        print("{0} - Investment Gain: {1}".format(self.project_name, self.investment_gain))
        print("{0} - ROI: {1}".format(self.project_name, self.roi))
        print("{0} - Annualized ROI: {1}".format(self.project_name, self.annualized_roi))

    def save_results_roi_measures(self):
        roi_measures = pd.DataFrame({c.PROJECT: [self.project_name],
                      c.MODEL: [c.MODULE],
                      c.AMOUNT_INVESTED: [self.amount_invested],
                      c.AMOUNT_RETURNED: [self.amount_returned],
                      c.ROI: [self.roi],
                      c.ANNUALIZED_ROI: [self.annualized_roi]})

        self.roi_measures = pd.concat([self.roi_measures, roi_measures])
        self.roi_measures.to_csv(self.roi_measures_file, header=False, mode = 'a', index=False)


project_list = ["angular/angular.js","ansible/ansible"]

for p in project_list:
  estimator = iROIEstimator(p)
  estimator.execute()
