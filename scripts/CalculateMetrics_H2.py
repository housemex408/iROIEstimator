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
def extractPerfMeasures(y_test, predictions, results):
  # r_squared = round(model.rsquared, 2)
  # r_squared_adj = round(model.rsquared_adj, 2)
  mae = round(metrics.mean_absolute_error(y_test, predictions), 2)
  mse = round(metrics.mean_squared_error(y_test, predictions), 2)
  rmse = round(np.sqrt(metrics.mean_squared_error(y_test, predictions)), 2)
  pred25 = round(utils.calculate_PRED(0.25, results), 2)
  pred50 = round(utils.calculate_PRED(0.50, results), 2)
  # return r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50
  return mae, mse, rmse, pred25, pred50


# def createDF(project_name, model, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records):
def createDF(project_name, model, task, mae, mse, rmse, pred25, pred50, t_records):
  row_df = pd.DataFrame({c.PROJECT: [project_name],
                      c.MODEL: [model],
                      c.TASK: [task],
                      # c.R_SQUARED: [r_squared],
                      # c.R_SQUARED_ADJ: [r_squared_adj],
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
directoryPath = "scripts/exports"
outputFile = "scripts/notebook/results/calculate_metrics_h2.csv".format(directory=directoryPath)
# headers = [c.PROJECT, c.MODEL, c.TASK, c.R_SQUARED, c.R_SQUARED_ADJ, c.MAE, c.MSE, c.RMSE, c.PRED_25, c.PRED_50, c.T_RECORDS]
headers = [c.PROJECT, c.MODEL, c.TASK, c.MAE, c.MSE, c.RMSE, c.PRED_25, c.PRED_50, c.T_RECORDS]
o_df = pd.DataFrame(columns=headers)

for project in c.PROJECT_LIST:
  project = project.split('/')[1]

  for task in c.TASK_LIST:

    tasks = "{directoryPath}/{project_name}/{project_name}_dataset_{task}.csv".format(directoryPath=directoryPath, project_name=project, task = task)

    # BEGIN Core Contributors
    df = pd.read_csv(tasks)
    df[c.DATE] = pd.to_datetime(df[c.DATE])
    df = df.dropna(subset=[c.TASK])
    df.fillna(df.mean(), inplace=True)
    if df.isna().values.any():
      df.fillna(0, inplace=True)
    # t_records = df.size

    df = utils.remove_outlier(df, c.LINE_CC)
    df = utils.remove_outlier(df, c.MODULE_CC)
    df = utils.remove_outlier(df, c.LINE_EC)
    df = utils.remove_outlier(df, c.MODULE_EC)

    t_records = df.size

    # Edge case when < 2 tasks detected
    if t_records < 2:
        break

    # Let's create multiple regression
    X_cc = df[[c.NT_CC, c.NO_CC]]
    Y_cc = df[c.LINE_CC]
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(X_cc, Y_cc, train_size=0.75, test_size=0.25, random_state=0)

    model = lm.OLS(y_train_cc, X_train_cc).fit()
    predictions_cc = model.predict(X_test_cc)

    X_ec = df[[c.NT_EC, c.NO_EC]]
    Y_ec = df[c.LINE_EC]
    X_train_ec, X_test_ec, y_train_ec, y_test_ec = train_test_split(X_ec, Y_ec, train_size=0.75, test_size=0.25, random_state=0)

    model = lm.OLS(y_train_ec, X_train_ec).fit()
    predictions_ec = model.predict(X_test_ec)
  
    y_test = y_test_cc + y_test_ec
    predictions = predictions_cc + predictions_ec
    results = compareResults(y_test, predictions)
    print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.LINE))

    
    # r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(y_test, predictions, results)
    mae, mse, rmse, pred25, pred50 = extractPerfMeasures(y_test, predictions, results)
    # row_df = createDF(project, c.LINE, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records)
    row_df = createDF(project, c.LINE, task, mae, mse, rmse, pred25, pred50, t_records)
    o_df = pd.concat([row_df, o_df])

    # Let's create multiple regression
    X_cc = df[[c.NT_CC, c.NO_CC]]
    Y_cc = df[c.MODULE_CC]
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(X_cc, Y_cc, train_size=0.75, test_size=0.25, random_state=0)

    model = lm.OLS(y_train_cc, X_train_cc).fit()
    predictions_cc = model.predict(X_test_cc)

    X_ec = df[[c.NT_EC, c.NO_EC]]
    Y_ec = df[c.MODULE_EC]
    X_train_ec, X_test_ec, y_train_ec, y_test_ec = train_test_split(X_ec, Y_ec, train_size=0.75, test_size=0.25, random_state=0)

    model = lm.OLS(y_train_ec, X_train_ec).fit()
    predictions_ec = model.predict(X_test_ec)
  
    y_test = y_test_cc + y_test_ec
    predictions = predictions_cc + predictions_ec
    results = compareResults(y_test, predictions)
    print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.MODULE))

    
    # r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(y_test, predictions, results)
    mae, mse, rmse, pred25, pred50 = extractPerfMeasures(y_test, predictions, results)
    # row_df = createDF(project, c.LINE, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records)
    row_df = createDF(project, c.MODULE, task, mae, mse, rmse, pred25, pred50, t_records)
    o_df = pd.concat([row_df, o_df])

    # END Core Contributors

o_df.sort_values(by=[c.PROJECT, c.MODEL, c.TASK], inplace=True)
print(tabulate(o_df, headers=headers))
o_df.to_csv(outputFile)

# END Main
