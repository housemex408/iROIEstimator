
import pandas as pd

def calculate_PRED(percentage, dataFrame):
    countLessPercent = dataFrame[dataFrame['Percent Error'] < percentage]['Percent Error']
    # print(countLessPercent.count())
    # print(dataFrame['Percent Error'].count())
    pred = countLessPercent.count() / dataFrame['Percent Error'].count()
    return pred

def say_hello():
    print("hello!")

