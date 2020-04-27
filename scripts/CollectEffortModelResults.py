from IPython import get_ipython
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import statsmodels as sm
import statsmodels.api as smapi
import statsmodels.regression.linear_model as lm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
sys.path.append(os.path.abspath("/Users/alvaradojo/Documents/Github/iROIEstimator/scripts"))
import Utilities as utils
import Constants as c

project_name = "angular"
directoryPath = "scripts/exports"
task_list = c.TASK_LIST
o_df = pd.DataFrame(
  columns=[c.PROJECT, c.MODEL, c.TASK, c.R_SQUARED, c.R_SQUARED_ADJ, c.MAE, c.MSE, c.RMSE, c.PRED_25, c.PRED_50]
)

for task in task_list:

    tasks = "{directoryPath}/{project_name}/{project_name}_dataset_{task}.csv".format(directoryPath=directoryPath, project_name=project_name, task = task)

    cc_columns = [c.VERSION, c.DATE, c.NT_CC, c.NO_CC, c.MODULE_CC, c.LINE_CC, c.T_MODULE, c.T_LINE, c.T_CC]
    df = pd.read_csv(tasks, usecols = cc_columns)
    df[c.DATE] = pd.to_datetime(df[c.DATE])
    df = df.dropna(subset=[c.T_MODULE])

    # Let's create multiple regression
    X = df[[c.NT_CC, c.NO_CC]]
    Y = df[c.LINE_CC]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=0)

    model = lm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)

    data = {c.OBSERVED:y_test, c.PREDICTED:predictions.round(2), c.DIFFERENCE:abs(y_test - predictions).round(2), c.PERCENT_ERROR:(abs(y_test - predictions)/y_test).round(2)}
    results = pd.DataFrame(data)

    r_squared = model.rsquared
    r_squared_adj = model.rsquared_adj
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    pred25 = round(utils.calculate_PRED(0.25, results), 2)
    pred50 = round(utils.calculate_PRED(0.50, results), 2)

    print("\n{0} - {1} - {2} model performance: \n".format(project_name, task, c.LINE_CC))
    # print(utils.format_perf_metric('Model - R Squared', r_squared))
    # print(utils.format_perf_metric('Model - R Squared Adj', r_squared_adj))
    # print(utils.format_perf_metric('Pred - Mean Absolute Error', mae))
    # print(utils.format_perf_metric('Pred - Mean Squared Error', mse))
    # print(utils.format_perf_metric('Pred - Root Mean Squared Error', rmse))
    # print(utils.format_PRED("25", pred25))
    # print(utils.format_PRED("50", pred50))

    row_df = pd.DataFrame({c.PROJECT: [project_name],
                        c.MODEL: [c.LINE_CC],
                        c.TASK: [task],
                        c.R_SQUARED: [r_squared],
                        c.R_SQUARED_ADJ: [r_squared_adj],
                        c.MAE: [mae],
                        c.MSE: [mse],
                        c.RMSE: [rmse],
                        c.PRED_25: [pred25],
                        c.PRED_50: [pred50]})

    o_df = pd.concat([row_df, o_df])
    print(o_df.head())

    # Let's create multiple regression
    X = df[[c.NT_CC, c.NO_CC, c.T_MODULE]]
    Y = df[c.MODULE_CC]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=0)
    model = lm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)

    data = {c.OBSERVED:y_test, c.PREDICTED:predictions.round(2), c.DIFFERENCE:abs(y_test - predictions).round(2), c.PERCENT_ERROR:(abs(y_test - predictions)/y_test).round(2)}
    results = pd.DataFrame(data)

    r_squared = model.rsquared
    r_squared_adj = model.rsquared_adj
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    pred25 = round(utils.calculate_PRED(0.25, results), 2)
    pred50 = round(utils.calculate_PRED(0.50, results), 2)

    print("\n{0} - {1} - {2} model performance: \n".format(project_name, task, c.MODULE_CC))
    # print(utils.format_perf_metric('Model - R Squared', r_squared))
    # print(utils.format_perf_metric('Model - R Squared Adj', r_squared_adj))
    # print(utils.format_perf_metric('Pred - Mean Absolute Error', mae))
    # print(utils.format_perf_metric('Pred - Mean Squared Error', mse))
    # print(utils.format_perf_metric('Pred - Root Mean Squared Error', rmse))
    # print(utils.format_PRED("25", pred25))
    # print(utils.format_PRED("50", pred50))

    row_df = pd.DataFrame({c.PROJECT: [project_name],
                        c.MODEL: [c.MODULE_CC],
                        c.TASK: [task],
                        c.R_SQUARED: [r_squared],
                        c.R_SQUARED_ADJ: [r_squared_adj],
                        c.MAE: [mae],
                        c.MSE: [mse],
                        c.RMSE: [rmse],
                        c.PRED_25: [pred25],
                        c.PRED_50: [pred50]})

    o_df = pd.concat([row_df, o_df])
    print(o_df.head())
