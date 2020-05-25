import re
import numpy as np
import pandas as pd
import logging
import logging.handlers
from scipy.stats import ttest_1samp
from scipy.stats import shapiro
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats import weightstats as stests
import os
import sys
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
from sklearn import preprocessing

def hot_encode(df, field):
    encoded_columns = pd.get_dummies(df[field])
    # print(encoded_columns)
    df = pd.concat([df,encoded_columns], axis=1)
    df.drop([field],axis=1, inplace=True)
    return df

def log_transform(df, field):
    return np.log1p(df[field])
    # return (df[field]+1).transform(np.log)

def reverse_log_transform(df):
    return np.expm1(df)

def normalize(df, field):
    return (df[field] - df[field].min()) / (df[field].max() - df[field].min())

def standardize(df, field):
    # data = np.reshape(df[field], (1, df[field].size))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(data)
    # n_test = pd.DataFrame(x_scaled, columns=[field])
    return
    return (df[field] - df[field].mean()) / df[field].std()

def calculate_PRED(percentage, dataFrame, percent_error_key):
    countLessPercent = dataFrame[dataFrame[percent_error_key] < percentage][percent_error_key]
    pred = countLessPercent.count() / dataFrame[percent_error_key].count()
    return pred

def format_PRED(percentage, value):
    return "Pred - PRED ({0}): {1:.2%}".format(percentage, value)

def format_perf_metric(label, value):
    return "{0}: {1}".format(label, round(value, 2))

def isRegularVersion(df):
    version_regex = r"^[v]{0,1}\d{1,2}\.\d{1,2}[\.\d{1,2}]{0,2}[\.\d{1,2}]{0,2}$"
    result = df[df[c.VERSION].str.match(version_regex)== True]
    return result

# https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def calculated_rsquared_adj(X, X_test, rsquared):
    k = len(X.columns)
    n = len(X_test)
    if ((n - k) == 1):
      return rsquared
    rsquared_adj = 1 - (((1-rsquared)*(n-1))/(n-k-1))
    return rsquared_adj

def get_logger():
  logger = logging.getLogger("iROIEstimator")
  if not logger.handlers:
    handler = logging.FileHandler('iROIEstimator.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
  return logger

def is_all_same(s):
  s_mean = s.mean()
  all_same = True
  for index, value in s.items():
    if value != s_mean:
      all_same = False
  return all_same

def percentage_nan(df):
  nans = df.size - df.count().sum()
  all = df.count().sum() + nans
  p_na = round(1 - (all - nans) / all, 3)
  return p_na

def gaussian_test(data, alpha):
  stat, p = shapiro(data)
  print("Shapiro p-value: ", round(p, 4))

  is_gaussian = True

  if p > alpha:
      print('Shapiro Test: Sample looks Gaussian (fail to reject H0)')
  else:
      is_gaussian = False
      print('Shapiro Test: Sample does not look Gaussian (reject H0)')

  return is_gaussian

def one_sample_t_test(data, mean, alpha):
  model_records_mean = round(data.mean(),2)

  ttest_result = ttest_1samp(data, mean)
  print("One Sample T-test p-value: ", round(ttest_result.pvalue / 2, 4))

  if ttest_result.pvalue / 2 > alpha:
      print("One Sample T-Test: {0} sample mean is likely to be greater than {1} (fail to reject H0)".format(model_records_mean, mean))
  else:
      print("One Sample T-Test: {0} sample mean is not likely to be greater than {1} (reject H0)".format(model_records_mean, mean))

def one_sample_z_test(data, mean, alpha):
  model_records_mean = round(data.mean(),2)

  ztest_result = stests.ztest(data, x2=None, value=mean)[1]
  print("One Sample Z-test p-value: ", round(ztest_result / 2, 4))

  if ztest_result / 2 > alpha:
      print("One Sample Z-Test: {0} sample mean is likely to be greater than {1} (fail to reject H0)".format(model_records_mean, mean))
  else:
      print("One Sample Z-Test: {0} sample mean is not likely to be greater than {1} (reject H0)".format(model_records_mean, mean))

def one_sample_sign_test(data, mean, alpha):
  model_records_mean = round(data.mean(),2)

  sign_test_result  = sign_test(data, mean)[1]
  print("One Sample Sign Test p-value: ", sign_test_result)

  if sign_test_result / 2 > alpha:
      print("One Sample Sign Test: {0} sample median is likely to be greater than {1} (fail to reject H0)".format(model_records_mean, mean))
  else:
      print("One Sample Sign Test: {0} sample median is not likely to be greater than {1} (reject H0)".format(model_records_mean, mean))
