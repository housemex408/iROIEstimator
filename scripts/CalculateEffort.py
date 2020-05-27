# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import os as os
import sys as sys
sys.path.append(os.path.abspath(__file__))
import Constants as c
import Utilities as utils
from Scaler import Scaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

def compareResults(Y, predictions):
  data = {}
  data[c.OBSERVED] = Y.round(2)
  data[c.PREDICTED] = predictions.round(2)
  data[c.DIFFERENCE] = abs(Y - predictions).round(2)
  data[c.PERCENT_ERROR] = (abs(Y - predictions)/Y).round(2)
  results = pd.DataFrame(data)
  results[c.PERCENT_ERROR].fillna(0, inplace=True)
  return results

directoryPath = "scripts/exports"
project_name = "angular"
task = "BUG"
file = "{0}/{1}/{2}_dataset_{3}.csv"

df = pd.read_csv(file.format(directoryPath, project_name, project_name, task))
# df = utils.isRegularVersion(df)
print(os.getcwd())


df[c.NT] = df[c.NT_CC] + df[c.NT_EC] + df[c.NT_UC]
df[c.NO] = df[c.NO_CC] + df[c.NO_EC] + df[c.NO_UC]
df[c.LINE] = df[c.LINE_CC] + df[c.LINE_EC] + df[c.LINE_UC]
df[c.MODULE] = df[c.MODULE_CC] + df[c.MODULE_EC] + df[c.MODULE_UC]
df[c.T_CONTRIBUTORS] = df[c.T_CC] + df[c.T_EC] + df[c.T_UC] + 2
# df.drop(df.columns[0], axis=1, inplace=True)
# df.drop(c.DATE, axis=1, inplace=True)
# df.drop(c.PROJECT, axis=1, inplace=True)

if df.isna().values.any():
    df.fillna(0, inplace=True)

X = df[[c.NT, c.NO, c.T_CONTRIBUTORS]]
Y = df[c.LINE]

splits = 10
num_records = len(X)

if num_records <= splits:
    splits = num_records

pipeline = Pipeline(steps=[('scaler', RobustScaler()), ('predictor', DecisionTreeRegressor(random_state=0))])
model = TransformedTargetRegressor(regressor=pipeline, transformer=RobustScaler())
model.fit(X, Y)

kfold = model_selection.KFold(n_splits=splits)
predictions = cross_val_predict(model, X, Y, cv=kfold)

results = compareResults(Y, predictions)
print(round(utils.calculate_PRED(0.25, results, c.PERCENT_ERROR), 2))
print(round(utils.calculate_PRED(0.50, results, c.PERCENT_ERROR), 2))
results.head()

# Let's create multiple regression
X = df[[c.NT, c.NO, c.T_CONTRIBUTORS]]
Y = df[c.MODULE]

splits = 10
num_records = len(X)

pipeline = Pipeline(steps=[('scaler', RobustScaler()), ('predictor', DecisionTreeRegressor(random_state=0))])
model = TransformedTargetRegressor(regressor=pipeline, transformer=RobustScaler())
model.fit(X,Y)
kfold = model_selection.KFold(n_splits=splits)
predictions = cross_val_predict(model, X, Y, cv=kfold)

results = compareResults(Y, predictions)
print(round(utils.calculate_PRED(0.25, results, c.PERCENT_ERROR), 2))
print(round(utils.calculate_PRED(0.50, results, c.PERCENT_ERROR), 2))
results.head()





