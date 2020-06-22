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
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR
import argparse
os.environ["NUMEXPR_MAX_THREADS"] = "12"

regressors = {
  "DecisionTreeRegressor": DecisionTreeRegressor(random_state=0, max_depth=10, min_samples_split=10),
  "RandomForestRegressor": RandomForestRegressor(random_state=0, max_depth=10, min_samples_split=10, n_estimators=10),
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
  "QuantileTransformer": QuantileTransformer(),
  "FunctionTransformer": FunctionTransformer(np.log1p)
}

regressor = regressors["DecisionTreeRegressor"]
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

def calculate_effort(X, Y, project, task, model_type, transformer, regressor, i_records, t_records):

  dummy_df = X.copy()
  dummy_df["Y"] = Y
  p_na = utils.percentage_nan(X)

  X.fillna(0, inplace=True)
  Y.fillna(0, inplace=True)

  # Let's create multiple regression
  print("\n{0} - {1} - {2} model performance: \n".format(project, task, model_type))

  splits = 10
  num_records = len(X)

  if num_records <= splits:
    splits = num_records

  pipeline = Pipeline(steps=[('scaler', transformer), ('predictor', regressor)])
  model = TransformedTargetRegressor(regressor=pipeline, transformer=transformer)
  model.fit(X, Y)

  kfold = model_selection.KFold(n_splits=splits)
  predictions = cross_val_predict(model, X, Y, cv=kfold)
  results = utils.create_percent_error_df(Y, predictions)

  r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50 = extractPerfMeasures(model, Y, predictions, results, X)
  row = createDF(project, model_type, task, r_squared, r_squared_adj, mae, mse, rmse, pred25, pred50, t_records, i_records - t_records, p_na)

  return row
# END Functions


# BEGIN Main
directoryPath = "scripts/exports"
outputFile = "scripts/notebook/results/h2_metrics_h1_DT_06_20_2020.csv".format(directory=directoryPath)
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
  df[c.T_LINE_P] = df[c.T_LINE].shift()

  # df[c.T_CONTRIBUTORS] = df[c.T_CONTRIBUTORS] + 2

  t_records = len(df)

  # Edge case when < 2 tasks detected
  if t_records < 2:
      continue

  # Calculate LINE
  line_cc_output = calculate_effort(df[[c.NT_CC, c.NO_CC, c.T_CC, c.T_LINE_P]], df[c.LINE_CC], project, task, c.LINE_CC, transformer, regressor, i_records, t_records)
  line_ec_output = calculate_effort(df[[c.NT_EC, c.NO_EC, c.T_EC, c.T_LINE_P]], df[c.LINE_EC], project, task, c.LINE_EC, transformer, regressor, i_records, t_records)
  line_uc_output = calculate_effort(df[[c.NT_UC, c.NO_UC, c.T_UC, c.T_LINE_P]], df[c.LINE_UC], project, task, c.LINE_UC, transformer, regressor, i_records, t_records)

  # Calculate MODULE
  module_cc_output = calculate_effort(df[[c.NT_CC, c.NO_CC, c.T_CC, c.T_LINE_P]], df[c.MODULE_CC], project, task, c.MODULE_CC, transformer, regressor, i_records, t_records)
  module_ec_output = calculate_effort(df[[c.NT_EC, c.NO_EC, c.T_EC, c.T_LINE_P]], df[c.MODULE_EC], project, task, c.MODULE_EC, transformer, regressor, i_records, t_records)
  module_uc_output = calculate_effort(df[[c.NT_UC, c.NO_UC, c.T_UC, c.T_LINE_P]], df[c.MODULE_UC], project, task, c.MODULE_UC, transformer, regressor, i_records, t_records)

  output = pd.concat([line_cc_output, line_ec_output, line_uc_output, module_cc_output, module_ec_output, module_uc_output])

  # Write to file
  output.sort_values(by=[c.PROJECT, c.MODEL, c.TASK], inplace=True)
  print(tabulate(output, headers=headers))
  output.to_csv(outputFile, header=False, mode = 'a', index=False)

# END Main
