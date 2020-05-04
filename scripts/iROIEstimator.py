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
    cwd = "scripts/exports"

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
        self.amount_invested = None
        self.amount_returned = None
        self.investment_gain = None
        self.roi = None
        self.annualized_roi = None

    
    def execute(self):
        for task in c.TASK_LIST:
            tasks = self.file_template.format(cwd=self.cwd, project_name=self.project_name, task = task)
            df = pd.read_csv(tasks)
            
            df = df.dropna(subset=[c.T_MODULE])
            df = df.dropna(subset=[c.DATE])
            df = df.dropna(subset=[c.TASK])
            df[c.DATE] = pd.to_datetime(df[c.DATE])
            df = df.set_index(c.DATE)
            df.index = df.index.strftime('%Y-%m-%d') 
            if df.isna().values.any():
                df.fillna(0, inplace=True)

            t_records = len(df)

            # Edge case when < 2 tasks detected
            if t_records < 2:
                break
            
            self.predict_effort(task, df)
            self.forecast_effort(df, task)
            # self.display_forecast(self.prediction_years)
            self.calculate_ROI()
    
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
        print("\n{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, c.MODULE_CC, module_cc_results.size))
        self.module_cc.calculate_perf_measurements()
        module_cc_output = self.module_cc.create_output_df()
        print(module_cc_output.head())

        # MODULE_EC
        self.module_ec = Effort(self.project_name, c.MODULE_EC, task, df)
        module_ec_results = self.module_ec.predict_effort()
        print("\n{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, c.MODULE_EC, module_ec_results.size))
        self.module_ec.calculate_perf_measurements()
        module_ec_output = self.module_ec.create_output_df()
        print(module_ec_output.head())

    def forecast_effort(self, df, task):
        self.module_cc.forecast_module_effort(self.predicton_months)
        self.module_ec.forecast_module_effort(self.predicton_months)

        self.task_forecasted_effort[task] = {
            c.MODULE_CC: self.module_cc.calculate_total_effort(self.prediction_years), 
            c.MODULE_EC: self.module_ec.calculate_total_effort(self.prediction_years)
        }

    def display_forecast(self, predictionYears):
        self.module_cc.display_forecast(predictionYears)
        self.module_ec.display_forecast(predictionYears)
    

    def calculate_ROI(self):
        t_effort_cc = 0
        t_effort_ec = 0

        for key in self.task_forecasted_effort:
            t_effort_cc = t_effort_cc + self.task_forecasted_effort[key][c.MODULE_CC].values.sum()
            t_effort_ec = t_effort_ec + self.task_forecasted_effort[key][c.MODULE_EC].values.sum()

        print("{0} - Core Contributor Forecasted Effort Over {1} years: {2}".format(self.project_name, self.prediction_years, round(t_effort_cc, 2)))
        print("{0} - External Contributor Forecasted Effort Over {1} years: {2}".format(self.project_name, self.prediction_years, round(t_effort_ec, 2)))

    # def printLine():

angular = iROIEstimator("angular/angular.js")
angular.execute()