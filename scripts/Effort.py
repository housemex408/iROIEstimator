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
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import concurrent.futures

class Effort:
    logger = utils.get_logger()

    def __init__(self, project, model, value, task, df, hourly_wage):
        self.modelType = model
        self.type = value
        self.task = task
        self.df = df
        self.t_records = len(self.df)
        self.project_name = project
        self.model = None
        self.predictions = None
        self.r_squared = None
        self.r_squared_adj = None
        self.mae = None
        self.mse = None
        self.rmse = None
        self.pred25 = None
        self.pred50 = None
        self.X = None
        self.Y = None
        self.results = None
        self.module_forecast_results = None
        self.average_effort_release = None
        self.hourly_wage = hourly_wage
        self.df = self.calculate_costs(df)
        # self.df = df

    def get_cost_columns(self):
      EFFORT = self.type
      T_CONTRIBS = None
      BILLED = None
      COST = c.COST
      HOURS_DIFF = None
      AVG_EFFORT_CONTRIBS = None
      CONTRIB_DIFF = None

      if self.type == c.LINE_CC or self.type == c.MODULE_CC:
        T_CONTRIBS = c.T_CC
        BILLED = c.BILLED_HOURS_CC
        HOURS_DIFF = c.HOURS_DIFF_CC
        AVG_EFFORT_CONTRIBS = c.AVG_MODULE_CONTRIBS_CC
        CONTRIB_DIFF = c.CONTRIB_DIFF_CC
      else:
        T_CONTRIBS = c.T_EC
        BILLED = c.BILLED_HOURS_EC
        HOURS_DIFF = c.HOURS_DIFF_EC
        AVG_EFFORT_CONTRIBS = c.AVG_MODULE_CONTRIBS_EC
        CONTRIB_DIFF = c.CONTRIB_DIFF_EC

      return EFFORT, T_CONTRIBS, BILLED, COST, HOURS_DIFF, AVG_EFFORT_CONTRIBS, CONTRIB_DIFF


    def minContrib(self, row, effort, contribs):
      if row[contribs] == 0 and row[effort] > 0:
          return 1
      else:
          return row[contribs]

    def calculate_costs(self, df):
      EFFORT, T_CONTRIBS, BILLED, COST, HOURS_DIFF, AVG_EFFORT_CONTRIBS, CONTRIB_DIFF = self.get_cost_columns()

      df[c.DATE_P] = df[c.DATE].shift()
      df[c.DATE_P].fillna(df[c.DATE].min(), inplace=True)

      # Cost section
      df[HOURS_DIFF] = utils.calculate_hours_diff(df)

      if df[[c.DATE, c.DATE_P]].isna().values.any():
        df[[c.DATE, c.DATE_P]].fillna(0, inplace=True)

      df[T_CONTRIBS] = df.apply(self.minContrib, effort=EFFORT, contribs=T_CONTRIBS, axis=1)

      average_effort = df[EFFORT].tail(30).mean()
      average_effort_contribs = df[T_CONTRIBS].mean()
      self.average_effort_release = average_effort / average_effort_contribs

      df[AVG_EFFORT_CONTRIBS] = df.apply(
        utils.calculate_contribs,
        effort=self.average_effort_release,
        model=EFFORT,
        contribs=T_CONTRIBS,
        axis=1
      )

      df[CONTRIB_DIFF] = round(df[T_CONTRIBS] - df[AVG_EFFORT_CONTRIBS], 2)
      df[BILLED] = round(df[HOURS_DIFF] * df[AVG_EFFORT_CONTRIBS], 2)
      df[COST] = round(df[BILLED] * self.hourly_wage, 2)

      return df

    def forecast_variable(self, variable, predicton_months):
        self.logger.debug("\n{0} - Forcasting {1} for {2} and task {3}: \n {4}".format(self.project_name, variable, self.type, self.task, self.df[variable]))

        data = {
            c.DATE: self.df[c.DATE],
            c.NT: self.df[variable]
        }
        NT = pd.DataFrame(data)
        NT.columns = ['ds','y']

        are_same = utils.is_all_same(NT['y'])

        if are_same:
          size = len(NT)
          min = NT['y'].head(1)
          NT['y'] = np.random.randint(min,min+2,size)

        NT['y_orig'] = NT['y']
        NT['y'], lam = boxcox(NT['y'] + 1)

        m_NT = Prophet(uncertainty_samples=0)
        m_NT.fit(NT)
        future_NT = m_NT.make_future_dataframe(periods = predicton_months, freq='m')
        forecast_NT = m_NT.predict(future_NT)

        # m_NT.plot(forecast_NT)

        forecast_NT_inv = pd.DataFrame()
        forecast_NT_inv['ds'] = forecast_NT['ds']
        # forecast_NT_inv[['yhat','yhat_upper','yhat_lower']] = forecast_NT[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))
        forecast_NT_inv[['yhat']] = forecast_NT[['yhat']].apply(lambda x: inv_boxcox(x, lam))

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

        # resultData = {variable: y_pred_rf.round(2), c.DATE: y_pred_index}
        data[variable] = y_pred_rf.round(2)
        data[c.DATE] = y_pred_index
        # results = pd.DataFrame(resultData)
        results = pd.DataFrame(data)
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

        pipeline = Pipeline(
          steps=[
            ('scaler', QuantileTransformer()),
            ('predictor', DecisionTreeRegressor(random_state=0, max_depth=10, min_samples_split=10))
          ])
        self.model = TransformedTargetRegressor(regressor=pipeline, transformer=QuantileTransformer())

        self.model.fit(self.X, self.Y)

        kfold = model_selection.KFold(n_splits=splits)
        self.predictions = cross_val_predict(self.model, self.X, self.Y, cv=kfold)

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

        EFFORT, T_CONTRIBS, BILLED, COST, HOURS_DIFF, AVG_EFFORT_CONTRIBS, CONTRIB_DIFF = self.get_cost_columns()

        data = {
            c.DATE: self.df[c.DATE],
            c.PROJECT: self.project_name,
            c.MODEL: TYPE,
            c.TASK: self.task,
            c.NT: self.X[NT],
            c.NO: self.X[NO],
            c.T_CONTRIBUTORS: self.X[T_CONTRIBUTORS],
            c.T_LINE: self.X[c.T_LINE_P],
            c.AVG_MODULE_CONTRIBS: self.df[AVG_EFFORT_CONTRIBS],
            c.HOURS_DIFF: self.df[HOURS_DIFF],
            c.CONTRIB_DIFF: self.df[CONTRIB_DIFF],
            c.BILLED_HOURS: self.df[BILLED],
            c.COST: self.df[COST]
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

    def forecast_module_effort(self, prediction_months, team_size=None):
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
            team_size = None

        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #   forecast_NT = executor.submit(self.forecast_variable, NT, predicton_months).result()
        #   forecast_NO = executor.submit(self.forecast_variable, NO, predicton_months).result()
        #   forecast_T_Contributors = executor.submit(self.forecast_variable, T_CONTRIBUTORS, predicton_months).result()
        #   forecast_T_Line_P = executor.submit(self.forecast_variable, c.T_LINE_P, predicton_months).result()
        forecast_NT = self.forecast_variable(NT, prediction_months)
        forecast_NO = self.forecast_variable(NO, prediction_months)
        forecast_T_Line_P = self.forecast_variable(c.T_LINE_P, prediction_months)
        forecast_T_Contributors = None

        if team_size != None:
          forecast_T_Contributors = {}
          df_size =  len(forecast_NT['yhat'])
          forecast_T_Contributors['yhat'] = utils.make_contrib_forecast(df_size, team_size)
        else:
          forecast_T_Contributors = self.forecast_variable(T_CONTRIBUTORS, prediction_months)

        data = {
            c.NT: forecast_NT['yhat'],
            c.NO: forecast_NO['yhat'],
            T_CONTRIBUTORS: forecast_T_Contributors['yhat'],
            c.T_LINE_P: forecast_T_Line_P['yhat']
        }

        dateIndex = forecast_NT['ds']
        self.module_forecast_results = self.forecast_effort(data, dateIndex, self.type, self.model)

        return self.module_forecast_results

    def calculate_total_effort(self, prediction_years):
        results = self.module_forecast_results
        results = self.calculate_costs(results)
        results['Year'] = results[c.DATE].apply(lambda x: x.year)
        results = pd.pivot_table(results,index=["Year"],values=[self.type, c.COST], aggfunc=np.sum).tail(prediction_years + 1)
        return results

    def display_forecast(self, prediction_years):
        results = self.calculate_total_effort(prediction_years)

        logger.info("\n{0} - {1} {2} Forecasted Effort: \n".format(self.project_name, self.type, self.task))
        logger.info(results.head(prediction_years + 1))
