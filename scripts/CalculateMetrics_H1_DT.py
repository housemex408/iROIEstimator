import numpy as np
import pandas as pd
import os
import sys
from sklearn import metrics
from tabulate import tabulate
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR
import argparse
os.environ["NUMEXPR_MAX_THREADS"] = "12"

regressors = {
  "DecisionTreeRegressor": DecisionTreeRegressor(random_state=0),
  "RandomForestRegressor": RandomForestRegressor(random_state=0),
  "AdaBoostRegressor": AdaBoostRegressor(random_state=0),
  "GradientBoostingRegressor": GradientBoostingRegressor(random_state=0),
  "ExtraTreesRegressor": ExtraTreesRegressor(random_state=0),
  "KNeighborsRegressor": KNeighborsRegressor(),
  "IsotonicRegression": IsotonicRegression(),
  "KernelRidge": KernelRidge(),
  "MLPRegressor": MLPRegressor(random_state=0),
  "SVR": SVR(),
  "LinearRegression": LinearRegression(),
}

transformers = {
  "RobustScaler": RobustScaler(),
  "StandardScaler": StandardScaler(),
  "MinMaxScaler": MinMaxScaler(),
  "QuantileTransformer": QuantileTransformer()
}

regressor = regressors["LinearRegression"]
transformer = transformers["QuantileTransformer"]

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
  results[c.PERCENT_ERROR].replace(np.inf, 0, inplace=True)
  return results

# END Functions


# BEGIN Main
directoryPath = "scripts/exports"
outputFile = "scripts/notebook/results/calculate_metrics_h1_LR_combined_05_27_2020.csv".format(directory=directoryPath)
headers = [c.PROJECT, c.MODEL, c.TASK, c.R_SQUARED, c.R_SQUARED_ADJ, c.MAE, c.MSE, c.RMSE, c.PRED_25, c.PRED_50, c.T_RECORDS, c.D_RECORDS, c.P_NA]
o_df = pd.DataFrame(columns=headers)


if not os.path.isfile(outputFile):
  o_df.to_csv(outputFile, index=False)

parser = argparse.ArgumentParser(description='Calculate Metrics')
parser.add_argument("--p")
args = parser.parse_args()
key = args.p[0:]
project = key.split('/')[1]
# project = "angular"

# for task in ["BUG"]:

for task in c.TASK_LIST:

  tasks = "{directoryPath}/{project_name}/{project_name}_dataset_{task}.csv".format(directoryPath=directoryPath, project_name=project, task = task)

  # BEGIN Core Contributors
  df = pd.read_csv(tasks)

  i_records = len(df)

  # df = utils.isRegularVersion(df)

  p_na = utils.percentage_nan(df)

  df[c.NT] = df[[c.NT_CC, c.NT_EC, c.NT_UC]].sum(axis=1)
  df[c.NO] = df[[c.NO_CC, c.NO_EC, c.NO_UC]].sum(axis=1)
  df[c.LINE] = df[[c.LINE_CC, c.LINE_EC, c.LINE_UC]].sum(axis=1)
  df[c.MODULE] = df[[c.MODULE_CC, c.MODULE_EC, c.MODULE_UC]].sum(axis=1)
  df[c.T_CONTRIBUTORS] = df[[c.T_CC, c.T_EC, c.T_UC]].sum(axis=1)
  df["T_LINE_DIFF"] = df[c.T_LINE].diff(-1)
  df["T_MODULE_DIFF"] = df[c.T_MODULE].diff(-1)
  df.dropna(subset=[c.MODULE, c.LINE, c.NT, c.NO, c.T_CONTRIBUTORS], inplace=True)

  if df.isna().values.any():
    df.fillna(0, inplace=True)

  df[c.T_CONTRIBUTORS] = df[c.T_CONTRIBUTORS] + 2

  t_records = len(df)

  # Edge case when < 2 tasks detected
  if t_records < 2:
      continue

  # Let's create multiple regression
  print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.LINE))

  X = df[[c.NT, c.NO, c.T_CONTRIBUTORS, "T_LINE_DIFF", "T_MODULE_DIFF"]]
  Y = df[c.LINE]
  splits = 10
  num_records = len(X)

  if num_records <= splits:
    splits = num_records

  pipeline = Pipeline(steps=[('scaler', transformer), ('predictor', regressor)])
  # pipeline = Pipeline(steps=[('scaler', RobustScaler()), ('predictor', DecisionTreeRegressor(random_state=0))])
  model = TransformedTargetRegressor(regressor=pipeline, transformer=transformer)
  model.fit(X, Y)

  kfold = model_selection.KFold(n_splits=splits)
  predictions = cross_val_predict(model, X, Y, cv=kfold)
  results = compareResults(Y, predictions)

  r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(model, Y, predictions, results, X)
  row_df_line = createDF(project, c.LINE, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records, i_records - t_records, p_na)

  # Let's create multiple regression
  print("\n{0} - {1} - {2} model performance: \n".format(project, task, c.MODULE))
  X = df[[c.NT, c.NO, c.T_CONTRIBUTORS, "T_LINE_DIFF", "T_MODULE_DIFF"]]
  Y = df[c.MODULE]

  # pipeline = Pipeline(steps=[('scaler', RobustScaler()), ('predictor', DecisionTreeRegressor(random_state=0))])
  pipeline = Pipeline(steps=[('scaler', transformer), ('predictor', regressor)])
  model = TransformedTargetRegressor(regressor=pipeline, transformer=transformer)
  model.fit(X, Y)
  # pipeline = Pipeline(steps=[('scaler', RobustScaler()), ('predictor', DecisionTreeRegressor(random_state=0))])

  # param_range = [1, 2, 3, 4, 5]

  # # Set grid search params
  # grid_params = [{
  #     'predictor__min_samples_leaf': param_range,
  #     'predictor__max_depth': param_range,
  #     'predictor__min_samples_split': param_range[1:]}]

  # model = TransformedTargetRegressor(
  #   regressor=GridSearchCV(estimator=pipeline, param_grid=grid_params, cv=5, n_jobs=-1),
  #   transformer=RobustScaler()
  # )

  # model = TransformedTargetRegressor(regressor=pipeline, transformer=RobustScaler())
  # model.fit(X, Y)

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
