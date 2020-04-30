import re
import pandas as pd

def calculate_PRED(percentage, dataFrame):
    countLessPercent = dataFrame[dataFrame['Percent Error'] < percentage]['Percent Error']
    # print(countLessPercent.count())
    # print(dataFrame['Percent Error'].count())
    pred = countLessPercent.count() / dataFrame['Percent Error'].count()
    return pred

def format_PRED(percentage, value):
    return "Pred - PRED ({0}): {1:.2%}".format(percentage, value)

def format_perf_metric(label, value):
    return "{0}: {1}".format(label, round(value, 2))

def isRegularVersion(value):
    result = re.findall(r"^[v]{0,1}\d{1,2}\.\d{1,2}\.\d{1,2}[\.\d{1,2}]{0,2}", value)
    if(result.__len__ == 0):
        return False
    return result[0] == value

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
    rsquared_adj = 1 - (((1-rsquared)*(n-1))/(n-k-1))
    return rsquared_adj

