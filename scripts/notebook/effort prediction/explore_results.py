# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
sys.path.append('../../')
import Constants as c


# %%
directoryPath = "../../exports"
file = "{directoryPath}/effort_prediction_performance.csv".format(directoryPath=directoryPath)
df = pd.read_csv(file)


# %%
df.drop(df.columns[0], axis=1)
df.dropna(subset=[c.PRED_25], inplace=True)


# %%
df.describe()


# %%
# t-test for task type

def hypothesisTest_Effort(model):
  print("t-test for: {0}".format(model))
  model_records = df[df[c.MODEL] == model]
  model_records_mean = model_records[c.PRED_25].mean()
  print(model_records_mean)
  ttest_result = ttest_1samp(model_records[c.PRED_25], 0.33)
  print("p-value: ", ttest_result.pvalue / 2)

  if ttest_result.pvalue / 2 < 0.10:
      print("Rejecting null hypothesis!")
  else:
      print("Accepting null hypothesis!")

hypothesisTest_Effort(c.LINE_CC)


# %%
print("t-test for: {0}".format(c.LINE_EC))
line_ec = df[df[c.MODEL] == c.LINE_EC]
line_ec_mean_pred50 = line_ec[c.PRED_25].mean()
print(line_ec_mean_pred50)
line_ec_result = ttest_1samp(line_ec_mean_pred50, 0.33)
print("p-values", line_ec_result)


# %%
print("t-test for: {0}".format(c.MODULE_CC))
module_cc = df[df[c.MODEL] == c.MODULE_CC]
module_cc_mean_pred50 = module_cc[c.PRED_25].mean()
print(module_cc_mean_pred50)
line_cc_result = ttest_1samp(module_cc_mean_pred50, 0.53)
print("p-values", line_cc_result)


# %%
print("t-test for: {0}".format(c.MODULE_EC))
module_ec = df[df[c.MODEL] == c.MODULE_EC]
module_ec_mean_pred50 = module_ec[c.PRED_25].mean()
print(module_ec_mean_pred50)
line_cc_result = ttest_1samp(module_ec_mean_pred50, 0.53)
print("p-values", line_cc_result)


