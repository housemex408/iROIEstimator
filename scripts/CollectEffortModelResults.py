from IPython import get_ipython
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import os
import sys
import statsmodels as sm
import statsmodels.api as smapi
import statsmodels.regression.linear_model as lm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tabulate import tabulate
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c

# BEGIN Functions
def extractPerfMeasures(model, y_test, predictions, results):
  r_squared = round(model.rsquared, 2)
  r_squared_adj = round(model.rsquared_adj, 2)
  mae = round(metrics.mean_absolute_error(y_test, predictions), 2)
  mse = round(metrics.mean_squared_error(y_test, predictions), 2)
  rmse = round(np.sqrt(metrics.mean_squared_error(y_test, predictions)), 2)
  pred25 = round(utils.calculate_PRED(0.25, results), 2)
  pred50 = round(utils.calculate_PRED(0.50, results), 2)
  # print(utils.format_perf_metric('Model - R Squared', r_squared))
  # print(utils.format_perf_metric('Model - R Squared Adj', r_squared_adj))
  # print(utils.format_perf_metric('Pred - Mean Absolute Error', mae))
  # print(utils.format_perf_metric('Pred - Mean Squared Error', mse))
  # print(utils.format_perf_metric('Pred - Root Mean Squared Error', rmse))
  # print(utils.format_PRED("25", pred25))
  # print(utils.format_PRED("50", pred50))
  return r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50


def createDF(project_name, model, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records):
  row_df = pd.DataFrame({c.PROJECT: [project_name],
                      c.MODEL: [model],
                      c.TASK: [task],
                      c.R_SQUARED: [r_squared],
                      c.R_SQUARED_ADJ: [r_squared_adj],
                      c.MAE: [mae],
                      c.MSE: [mse],
                      c.RMSE: [rmse],
                      c.PRED_25: [pred25],
                      c.PRED_50: [pred50],
                      c.T_RECORDS: t_records})
  return row_df


def compareResults(y_test, predictions):
  data = {c.OBSERVED:y_test, c.PREDICTED:predictions.round(2), c.DIFFERENCE:abs(y_test - predictions).round(2), c.PERCENT_ERROR:(abs(y_test - predictions)/y_test).round(2)}
  results = pd.DataFrame(data)
  return results

# END Functions


# BEGIN Main

project = "elasticsearch"
directoryPath = "scripts/exports"
outputFile = "{directory}/effort_prediction_performance.csv".format(directory=directoryPath)
headers = [c.PROJECT, c.MODEL, c.TASK, c.R_SQUARED, c.R_SQUARED_ADJ, c.MAE, c.MSE, c.RMSE, c.PRED_25, c.PRED_50, c.T_RECORDS]
o_df = pd.DataFrame(columns=headers)

for project in c.PROJECT_LIST:
  project = project.split('/')[1]

  for task in c.TASK_LIST:

    tasks = "{directoryPath}/{project_name}/{project_name}_dataset_{task}.csv".format(directoryPath=directoryPath, project_name=project, task = task)

    # BEGIN Core Contributors
    cc_columns = [c.TASK, c.VERSION, c.DATE, c.NT_CC, c.NO_CC, c.MODULE_CC, c.LINE_CC, c.T_MODULE, c.T_LINE, c.T_CC]
    df = pd.read_csv(tasks, usecols = cc_columns)
    df[c.DATE] = pd.to_datetime(df[c.DATE])
    df = df.dropna(subset=[c.TASK])
    df.fillna(df.mean(), inplace=True)
    if df.isna().values.any():
      df.fillna(0, inplace=True)
    t_records = df.size

    # Edge case when < 2 tasks detected
    if t_records < 2:
        break

    # Let's create multiple regression
    X = df[[c.NT_CC, c.NO_CC]]
    Y = df[c.LINE_CC]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=0)

    print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.LINE_CC))

    model = lm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)
    results = compareResults(y_test, predictions)

    r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(model, y_test, predictions, results)
    row_df = createDF(project, c.LINE_CC, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records)
    o_df = pd.concat([row_df, o_df])

    # Let's create multiple regression
    X = df[[c.NT_CC, c.NO_CC, c.T_MODULE]]
    Y = df[c.MODULE_CC]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=0)

    print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.MODULE_CC))

    model = lm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)
    results = compareResults(y_test, predictions)

    r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(model, y_test, predictions, results)
    row_df = createDF(project, c.MODULE_CC, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records)
    o_df = pd.concat([row_df, o_df])

    # END Core Contributors

    # BEGIN External Contributors
    ec_columns = [c.TASK, c.VERSION, c.DATE, c.NT_EC, c.NO_EC, c.MODULE_EC, c.LINE_EC, c.T_MODULE, c.T_LINE, c.T_EC]
    df = pd.read_csv(tasks, usecols = ec_columns)
    df[c.DATE] = pd.to_datetime(df[c.DATE])
    df = df.dropna(subset=[c.TASK])
    df.fillna(df.mean(), inplace=True)
    if df.isna().values.any():
      df.fillna(0, inplace=True)
    t_records = df.size

    # Let's create multiple regression
    X = df[[c.NT_EC, c.NO_EC]]
    Y = df[c.LINE_EC]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=0)

    print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.LINE_EC))

    model = lm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)
    results = compareResults(y_test, predictions)

    r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(model, y_test, predictions, results)
    row_df = createDF(project, c.LINE_EC, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records)
    o_df = pd.concat([row_df, o_df])

    # Let's create multiple regression
    X = df[[c.NT_EC, c.NO_EC, c.T_MODULE]]
    Y = df[c.MODULE_EC]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=0)

    print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.MODULE_EC))

    model = lm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)
    results = compareResults(y_test, predictions)

    r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(model, y_test, predictions, results)
    row_df = createDF(project, c.MODULE_EC, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records)
    o_df = pd.concat([row_df, o_df])

    # END External Contributors
o_df.sort_values(by=[c.PROJECT, c.MODEL, c.TASK], inplace=True)
print(tabulate(o_df, headers=headers))
# o_df.to_csv(outputFile)

# END Main
