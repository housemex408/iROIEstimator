import re
import numpy as np
import pandas as pd
import logging
import logging.handlers
from scipy.stats import ttest_1samp, wilcoxon, mannwhitneyu
from scipy.stats import shapiro, kruskal
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats import weightstats as stests
import os
import sys
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
from sklearn import preprocessing
from business_duration import businessDuration
import holidays as pyholidays
from datetime import time
from itertools import repeat

#Business open hour must be in standard python time format-Hour,Min,Sec
biz_open_time=time(8,0,0)

#Business close hour must be in standard python time format-Hour,Min,Sec
biz_close_time=time(17,0,0)

#US public holidays
US_holiday_list = pyholidays.US(state='CA')

#Business duration can be 'day', 'hour', 'min', 'sec'
unit_hour='hour'

#Weekend list. 5-Sat, 6-Sun
weekend_list = [5,6]

def calculate_hours_diff(df):
    return list(map(
        businessDuration, df[c.DATE_P], df[c.DATE], repeat(biz_open_time), repeat(biz_close_time), repeat(weekend_list), repeat(US_holiday_list), repeat(unit_hour)
    ))

def calculate_contribs(row, effort, model, contribs):
    if effort == 0:
      effort = 1

    contributors = row[model] / effort
    min_contribs = min(contributors, row[contribs])

    if min_contribs == 0:
        return 1

    return round(min_contribs, 2)

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
    return (df[field] - df[field].mean()) / df[field].std()

def calculate_PRED(percentage, dataFrame, percent_error_key):
    countLessPercent = dataFrame[dataFrame[percent_error_key] < percentage][percent_error_key]
    pred = countLessPercent.count() / dataFrame[percent_error_key].count()
    return round(pred, 2)

def percent_error(y, y_pred):
    if y == 0 and y_pred != 0:
        return abs(y_pred)
    elif y == 0 and y_pred == 0:
        return 0 

    error = abs((y - y_pred)/y)

    return round(error, 2)

def create_percent_error_df(y, y_pred):
    data = {}

    data[c.OBSERVED] = y.round(2)
    data[c.PREDICTED] = y_pred.round(2)
    data[c.DIFFERENCE] = abs(y - y_pred).round(2)
    data[c.PERCENT_ERROR] = np.vectorize(percent_error)(y, y_pred)
    
    results = pd.DataFrame(data)

    return results
    
def pred_25_scorer(estimator, X, y):
    y_pred = estimator.predict(X)

    results = create_percent_error_df(y, y_pred)

    pred = calculate_PRED(.25, results, c.PERCENT_ERROR)

    return round(pred, 2)

def pred_50_scorer(estimator, X, y):
    y_pred = estimator.predict(X)

    results = create_percent_error_df(y, y_pred)

    pred = calculate_PRED(.50, results, c.PERCENT_ERROR)

    return round(pred, 2)

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
  print("Shapiro p-value: ", p)

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
  print("One Sample T-test p-value: ", ttest_result.pvalue)

  if ttest_result.pvalue > alpha:
      print("One Sample T-Test: {0} sample mean is likely to be greater than {1} (fail to reject H0)".format(model_records_mean, mean))
  else:
      print("One Sample T-Test: {0} sample mean is not likely to be greater than {1} (reject H0)".format(model_records_mean, mean))

def one_sample_z_test(data, mean, alpha):
  model_records_mean = round(data.mean(),2)

  tstat, pvalue = stests.ztest(data, x2=None, value=mean, alternative='smaller')
  print("One Sample Z-test p-value: ", pvalue)

  if pvalue > alpha:
      print("One Sample Z-Test: {0} sample mean is likely to be greater than {1} (fail to reject H0)".format(model_records_mean, mean))
  else:
      print("One Sample Z-Test: {0} sample mean is not likely to be greater than {1} (reject H0)".format(model_records_mean, mean))

def one_sample_sign_test(data, mean, alpha):
    model_records_mean = round(data.mean(), 2)

    # pvalue  = sign_test(data, mean)[1]
    z_statistic, pvalue = wilcoxon(data - mean, alternative='less')
    print("One Sample Sign Test p-value: ", pvalue)

    if pvalue > alpha:
        print("One Sample Sign Test: {0} sample median is likely to be greater than {1} (fail to reject H0)".format(model_records_mean, mean))
    else:
        print("One Sample Sign Test: {0} sample median is not likely to be greater than {1} (reject H0)".format(model_records_mean, mean))

def two_sample_rank_test(s1, s2, model1, model2, alpha):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    # https://sixsigmastudyguide.com/mann-whitney-non-parametric-hypothesis-test/
    statistic, pvalue  = mannwhitneyu(s1, s2, alternative='two-sided')
    # z_statistic, pvalue = wilcoxon(data - mean, alternative='less')
    print("Two Sample Rank Test p-value: ", pvalue)

    if pvalue > alpha:
        print("Two Sample Rank Test: {0} median is likely to be the same {1} (fail to reject H0)".format(model1, model2))
    else:
        print("Two Sample Rank Test: {0} median is not likely to be the same as {1} (reject H0)".format(model1, model2))

def multi_sample_rank_test(s1, s2, s3, s4, s5, s6, s7, s8, s9, alpha):
    s1_median = round(s1.median(), 2)
    s2_median = round(s2.median(), 2)
    s3_median = round(s3.median(), 2)
    s4_median = round(s4.median(), 2)
    s5_median = round(s5.median(), 2)
    s6_median = round(s6.median(), 2)
    s7_median = round(s7.median(), 2)
    s8_median = round(s8.median(), 2)
    s9_median = round(s9.median(), 2)

    statistic, pvalue  = kruskal(s1, s2, s3, s4, s5, s6, s7, s8, s9)
    # z_statistic, pvalue = wilcoxon(data - mean, alternative='less')
    print("\nNon-parametric ANOVA p-value: ", pvalue)

    if pvalue > alpha:
        print("Non-parametric ANOVA: Distribution of all samples are likely to be the same (fail to reject H0)")
    else:
        print("Non-parametric ANOVA: At least one sample has a distribution different from the others (reject H0)")


