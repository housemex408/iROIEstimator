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
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
import argparse
from Scaler import Scaler
# os.environ["NUMEXPR_MAX_THREADS"] = "12"

# BEGIN Functions
def extractPerfMeasures(model, Y, predictions, results, X):
  r_squared = round(model.score(X, Y), 2)
  r_squared_adj = round(utils.calculated_rsquared_adj(X, X, r_squared), 2)
  mae = round(metrics.mean_absolute_error(Y, predictions), 2)
  mse = round(metrics.mean_squared_error(Y, predictions), 2)
  rmse = round(np.sqrt(metrics.mean_squared_error(Y, predictions)), 2)
  pred25 = round(utils.calculate_PRED(0.25, results, c.PERCENT_ERROR), 2)
  pred50 = round(utils.calculate_PRED(0.50, results, c.PERCENT_ERROR), 2)
  return r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50


def createDF(project_name, model, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records, d_records, p_na):
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
                      c.T_RECORDS: [t_records],
                      c.D_RECORDS: [d_records],
                      c.P_NA: [p_na]})
  return row_df


def compareResults(Y, predictions):
  data = {}
  data[c.OBSERVED] = Y.round(2)
  data[c.PREDICTED] = predictions.round(2)
  data[c.DIFFERENCE] = abs(Y - predictions).round(2)
  data[c.PERCENT_ERROR] = (abs(Y - predictions)/Y).round(2)
  results = pd.DataFrame(data)
  results[c.PERCENT_ERROR].fillna(0, inplace=True)
  return results

# END Functions


# BEGIN Main
directoryPath = "scripts/exports"
outputFile = "scripts/notebook/results/calculate_metrics_h1_DT_combined_05_26_2020.csv".format(directory=directoryPath)
headers = [c.PROJECT, c.MODEL, c.TASK, c.R_SQUARED, c.R_SQUARED_ADJ, c.MAE, c.MSE, c.RMSE, c.PRED_25, c.PRED_50, c.T_RECORDS, c.D_RECORDS, c.P_NA]
o_df = pd.DataFrame(columns=headers)

if not os.path.isfile(outputFile):
  o_df.to_csv(outputFile, index=False)

parser = argparse.ArgumentParser(description='Calculate Metrics')
parser.add_argument("--p")
args = parser.parse_args()
key = args.p[0:]
project = key.split('/')[1]


for task in c.TASK_LIST:

  tasks = "{directoryPath}/{project_name}/{project_name}_dataset_{task}.csv".format(directoryPath=directoryPath, project_name=project, task = task)

  # BEGIN Core Contributors
  df = pd.read_csv(tasks)

  i_records = len(df)

  # df = utils.isRegularVersion(df)

  p_na = utils.percentage_nan(df)

  if df.isna().values.any():
    df.fillna(0, inplace=True)

  df[c.NT] = df[c.NT_CC] + df[c.NT_EC] + df[c.NT_UC]
  df[c.NO] = df[c.NO_CC] + df[c.NO_EC] + df[c.NO_UC]
  df[c.LINE] = df[c.LINE_CC] + df[c.LINE_EC] + df[c.LINE_UC]
  df[c.MODULE] = df[c.MODULE_CC] + df[c.MODULE_EC] + df[c.MODULE_UC]
  df[c.T_CONTRIBUTORS] = df[c.T_CC] + df[c.T_EC] + df[c.T_UC] + 2

  NT_Scaler = Scaler(df[c.NT])
  NO_Scaler = Scaler(df[c.NO])
  T_C_Scaler = Scaler(df[c.T_CONTRIBUTORS])

  data = {
      c.NT: NT_Scaler.transform(),
      c.NO: NO_Scaler.transform(),
      c.T_CONTRIBUTORS: T_C_Scaler.transform()
  }
  df_scaled = pd.DataFrame(data)

  t_records = len(df)

  # Edge case when < 2 tasks detected
  if t_records < 2:
      continue

  # Let's create multiple regression
  print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.LINE))

  X = df_scaled[[c.NT, c.NO, c.T_CONTRIBUTORS]]
  Y = df[c.LINE]
  splits = 10
  num_records = len(X)

  if num_records <= splits:
    splits = num_records

  regress = DecisionTreeRegressor(random_state=0)
  model = TransformedTargetRegressor(regressor=regress,transformer=RobustScaler())
  model.fit(X, Y)
  kfold = model_selection.KFold(n_splits=splits)
  predictions = cross_val_predict(model, X, Y, cv=kfold)
  results = compareResults(Y, predictions)

  r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(model, Y, predictions, results, X)
  row_df_line = createDF(project, c.LINE, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records, i_records - t_records, p_na)


  # Let's create multiple regression
  print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.MODULE))
  X = df_scaled[[c.NT, c.NO, c.T_CONTRIBUTORS]]
  Y = df[c.MODULE]

  regress = DecisionTreeRegressor(random_state=0)
  model = TransformedTargetRegressor(regressor=regress,transformer=RobustScaler())
  model.fit(X, Y)
  kfold = model_selection.KFold(n_splits=splits)
  predictions = cross_val_predict(model, X, Y, cv=kfold)
  results = compareResults(Y, predictions)

  r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(model, Y, predictions, results, X)
  row_df_module = createDF(project, c.MODULE, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records, i_records - t_records, p_na)
  output = pd.concat([row_df_line, row_df_module])

  # END Core Contributors

  output.sort_values(by=[c.PROJECT, c.MODEL, c.TASK], inplace=True)
  print(tabulate(output, headers=headers))
  output.to_csv(outputFile, header=False, mode = 'a', index=False)

# END Main
