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
import random
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeRegressor

class Effort:
    logger = utils.get_logger()

    def __init__(self, project, model, value, task, df):
        self.modelType = model
        self.type = value
        self.task = task
        self.df = df
        self.t_records = len(self.df)
        self.project_name = project
        self.model = None
        self.predictions = None
        self.predictions_X = None
        self.r_squared = None
        self.r_squared_X = None
        self.r_squared_adj = None
        self.mae = None
        self.mse = None
        self.rmse = None
        self.pred25 = None
        self.pred50 = None
        self.pred25_X = None
        self.pred50_X = None
        self.predicted_effort = None
        self.shapiro_wilk_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.Y = None
        self.results = None
        self.results_x_validated = None
        self.module_forecast_results = None

    def forecast_variable(self, variable, predicton_months):
        self.logger.debug("\n{0} - Forcasting {1} for {2} and task {3}: \n {4}".format(self.project_name, variable, self.type, self.task, self.df[variable]))

        data = {
            c.DATE: self.df.index,
            c.NT: self.df[variable]
        }
        NT = pd.DataFrame(data)
        NT.columns = ['ds','y']

        are_same = utils.is_all_same(NT['y'])

        for i in range(len(NT)):
          y_value = NT['y'][i]
          if are_same or y_value == 0:
            NT['y'][i] = random.randint(1,4)

        NT['y_orig'] = NT['y']
        NT['y'], lam = boxcox(NT['y'])

        m_NT = Prophet(interval_width=0.90)
        m_NT.fit(NT)
        future_NT = m_NT.make_future_dataframe(periods = predicton_months, freq='m')
        forecast_NT = m_NT.predict(future_NT)

        # m_NT.plot(forecast_NT)

        forecast_NT_inv = pd.DataFrame()
        forecast_NT_inv['ds'] = forecast_NT['ds']
        forecast_NT_inv[['yhat','yhat_upper','yhat_lower']] = forecast_NT[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))

        m_NT.history['y_t'] = m_NT.history['y']
        m_NT.history['y'] = m_NT.history['y_orig']

        NT['y_t'] = NT['y']
        NT['y'] = NT['y_orig']

        # m_NT.plot(forecast_NT_inv)
        forecast_NT_inv['yhat'].fillna(forecast_NT_inv['yhat'].tail(predicton_months - 12).mean(), inplace=True)

        self.forecast = forecast_NT_inv

        return self.forecast

    def forecast_effort(self, data, dateIndex, variable, rf_regressor):
        X_Future = pd.DataFrame(data)

        X_Future = X_Future.astype('float32')

        X_Future = X_Future.replace([np.inf, -np.inf, np.nan], 0)

        y_pred_rf = rf_regressor.predict(X_Future)
        y_pred_index = dateIndex

        resultData = {variable: y_pred_rf.round(2), c.DATE: y_pred_index}
        results = pd.DataFrame(resultData)
        return results

    def predict_effort(self):
        self.df[c.T_LINE_P] = self.df[c.T_LINE].shift()

        if self.df.isna().values.any():
          self.df.fillna(0, inplace=True)

        if self.type == c.LINE_CC:
            self.X = self.df[[c.NT_CC, c.NO_CC, c.T_CC, c.T_LINE_P]]
            self.Y = self.df[c.LINE_CC]
        elif self.type == c.LINE_EC:
            self.X = self.df[[c.NT_EC, c.NO_EC, c.T_EC, c.T_LINE_P]]
            self.Y = self.df[c.LINE_EC]
        elif self.type == c.MODULE_CC:
            self.X = self.df[[c.NT_CC, c.NO_CC, c.T_CC, c.T_LINE_P]]
            self.Y = self.df[c.MODULE_CC]
        elif self.type == c.MODULE_EC:
            self.X = self.df[[c.NT_EC, c.NO_EC, c.T_EC, c.T_LINE_P]]
            self.Y = self.df[c.MODULE_EC]

        splits = 10

        if self.t_records <= splits:
          splits = self.t_records

        self.model = DecisionTreeRegressor(random_state=0)
        self.model.fit(self.X, self.Y)

        kfold = model_selection.KFold(n_splits=splits)
        self.predictions = cross_val_predict(self.model, self.X, self.Y, cv=kfold)

        # self.logger.info("{0} - {1} - {2} R2 Scores: {3}".format(self.project_name, self.task, self.type, results_kfold.round(2)))
        # self.logger.info("{0} - {1} - {2} R2 Accuracy: {3}".format(self.project_name, self.task, self.type, self.r_squared_X))
        # self.logger.info("{0} - {1} - {2} Prediction Count: {3}".format(self.project_name, self.task, self.type, len(self.Y)))
        # self.logger.info("{0} - {1} - {2} X-Validated Prediction Count: {3}".format(self.project_name, self.task, self.type, len(self.predictions_X)))

        results = self.calculate_diff()

        return results

    def calculate_diff(self):
        NT = None
        NO = None
        T_CONTRIBUTORS = None
        TYPE = self.type

        if self.type == c.LINE_CC or self.type == c.MODULE_CC:
            NT = c.NT_CC
            NO = c.NO_CC
            T_CONTRIBUTORS = c.T_CC
        elif self.type == c.LINE_EC or self.type == c.MODULE_EC:
            NT = c.NT_EC
            NO = c.NO_EC
            T_CONTRIBUTORS = c.T_EC

        data = {
            c.PROJECT: self.project_name,
            c.MODEL: TYPE,
            c.TASK: self.task,
            c.NT: self.X[NT],
            c.NO: self.X[NO],
            c.T_CONTRIBUTORS: self.X[T_CONTRIBUTORS],
            c.T_LINE_P: self.X[c.T_LINE_P]
        }

        data[c.OBSERVED] = self.Y.round(2)
        data[c.PREDICTED] = self.predictions.round(2)
        data[c.DIFFERENCE] = abs(self.Y - self.predictions).round(2)
        data[c.PERCENT_ERROR] = (abs(self.Y - self.predictions)/self.Y).round(2)

        self.results = pd.DataFrame(data)
        self.results[c.PERCENT_ERROR].fillna(0, inplace=True)
        self.results[c.PERCENT_ERROR].replace(np.inf, 0, inplace=True)

        return self.results

    def calculate_perf_measurements(self):
        self.r_squared = round(self.model.score(self.X, self.Y), 2)
        self.r_squared_adj = round(utils.calculated_rsquared_adj(self.X, self.X, self.r_squared), 2)
        self.mae = round(metrics.mean_absolute_error(self.Y, self.predictions), 2)
        self.mse = round(metrics.mean_squared_error(self.Y, self.predictions), 2)
        self.rmse = round(np.sqrt(metrics.mean_squared_error(self.Y, self.predictions)), 2)
        self.pred25 = round(utils.calculate_PRED(0.25, self.results, c.PERCENT_ERROR), 2)
        self.pred50 = round(utils.calculate_PRED(0.50, self.results, c.PERCENT_ERROR), 2)

    def create_output_df(self):
        row_df = pd.DataFrame({c.PROJECT: [self.project_name],
                      c.MODEL: [self.type],
                      c.TASK: [self.task],
                      c.R_SQUARED: [self.r_squared],
                      c.R_SQUARED_ADJ: [self.r_squared_adj],
                      c.MAE: [self.mae],
                      c.MSE: [self.mse],
                      c.RMSE: [self.rmse],
                      c.PRED_25: [self.pred25],
                      c.PRED_50: [self.pred50],
                      c.T_RECORDS: self.t_records})
        return row_df

    def forecast_module_effort(self, predicton_months):
        NT = None
        NO = None
        T_CONTRIBUTORS = None

        if self.type == c.LINE_CC or self.type == c.MODULE_CC:
            NT = c.NT_CC
            NO = c.NO_CC
            T_CONTRIBUTORS = c.T_CC
        elif self.type == c.LINE_EC or self.type == c.MODULE_EC:
            NT = c.NT_EC
            NO = c.NO_EC
            T_CONTRIBUTORS = c.T_EC

        forecast_NT = self.forecast_variable(NT, predicton_months)
        forecast_NO = self.forecast_variable(NO, predicton_months)
        forecast_T_Contributors = self.forecast_variable(T_CONTRIBUTORS, predicton_months)
        forecast_T_Line_P = self.forecast_variable(c.T_LINE_P, predicton_months)


        data = {
            c.NT: forecast_NT['yhat'],
            c.NO: forecast_NO['yhat'],
            c.T_CONTRIBUTORS: forecast_T_Contributors['yhat'],
            c.T_LINE_P: forecast_T_Line_P['yhat']
        }

        dateIndex = forecast_NT['ds']
        self.module_forecast_results = self.forecast_effort(data, dateIndex, self.type, self.model)

        return self.module_forecast_results

    def calculate_total_effort(self, prediction_years):
        results = self.module_forecast_results
        results['Year'] = results[c.DATE].apply(lambda x: x.year)
        results = pd.pivot_table(results,index=["Year"],values=[self.type], aggfunc=np.sum).tail(prediction_years + 1)
        return results

    def display_forecast(self, prediction_years):
        results = self.calculate_total_effort(prediction_years)

        logger.info("\n{0} - {1} {2} Forecasted Effort: \n".format(self.project_name, self.type, self.task))
        logger.info(results.head(prediction_years + 1))
