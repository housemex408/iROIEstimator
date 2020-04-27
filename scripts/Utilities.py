
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

