import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c

class EffortMeasurements:
    def __init__(self, value, task, df):
        self.type = value
        self.task = task
        self.df = df
        self.predictions = None
        self.mae = None
        self.mse = None
        self.rmse = None
        self.pred25 = None
        self.pred50 = None
        self.predicted_effort = None
        self.shapiro_wilk_test = None

    def predict_effort(self):
        X = None
        Y = None

        if self.type == c.LINE_CC:
            X = self.df[[c.NT_CC, c.NO_CC]]
            Y = self.df[c.LINE_CC]
        elif self.type == c.LINE_EC:
            X = self.df[[c.NT_EC, c.NO_EC]]
            Y = self.df[c.LINE_EC]
        elif self.type == c.MODULE_CC:
            X = self.df[[c.NT_CC, c.NO_CC]]
            Y = self.df[c.MODULE_CC]
        elif self.type == c.MODULE_EC:
            X = self.df[[c.NT_EC, c.NO_EC]]
            Y = self.df[c.MODULE_EC]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=0)
        model = RandomForestRegressor(n_estimators=300, random_state=0)
        model.fit(X_train, y_train)
        self.predictions = model.predict(X_test)

        return self.predictions

    def calculate_perf_measurements(self):
        return None



class iROIEstimator:
    cwd = "scripts/exports"

    def __init__(self, project):
        self.project_name = project.split('/')[1];
        self.file_template = "{cwd}/{project_name}/{project_name}_dataset_{task}.csv"    
    
    def predict_effort(self):
        for task in c.TASK_LIST:
            tasks = self.file_template.format(cwd=self.cwd, project_name=self.project_name, task = task)
            df = pd.read_csv(tasks)
            
            df[c.DATE] = pd.to_datetime(df[c.DATE])
            df = df.dropna(subset=[c.TASK])
            df.fillna(df.mean(), inplace=True)
            if df.isna().values.any():
                df.fillna(0, inplace=True)
            t_records = len(df)

            # Edge case when < 2 tasks detected
            if t_records < 2:
                break

            # # LINE_CC
            # line_cc = EffortMeasurements(c.LINE_CC, task, df)
            # line_cc_results = line_cc.predict_effort()
            # print("\n{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, c.LINE_CC, line_cc_results.size))

            # LINE_EC
            # line_ec = EffortMeasurements(c.LINE_EC, task, df)
            # line_ec_results = line_ec.predict_effort()
            # print("\n{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, c.LINE_EC, line_ec_results.size))

            # MODULE_CC
            # module_cc = EffortMeasurements(c.MODULE_CC, task, df)
            # module_cc_results = module_cc.predict_effort()
            # print("\n{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, c.MODULE_CC, module_cc_results.size))

            # MODULE_EC
            module_ec = EffortMeasurements(c.MODULE_EC, task, df)
            module_ec_results = module_ec.predict_effort()
            module_ec.calculate_perf_measurements()
            print("\n{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, c.MODULE_EC, module_ec_results.size))

            

    # def forecastEffort():

    # def calculateROI():

    # def printLine():

angular = iROIEstimator("angular/angular")
angular.predict_effort()