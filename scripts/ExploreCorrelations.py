import numpy as np
import pandas as pd
import os
import sys
from sklearn import metrics
from tabulate import tabulate
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
from scipy.stats import pearsonr
from scipy.stats import spearmanr
os.environ["NUMEXPR_MAX_THREADS"] = "12"


# BEGIN Main
directoryPath = "scripts/exports"
outputFile = "scripts/notebook/data_analysis/correlations/{project}_correlations.csv"

for project in c.ALL_PROJECTS:
# for project in ["angular"]:
  # for task in ["BUG"]:

  project = project.split('/')[1]
  correlationsFile = open(outputFile.format(project=project), "w")

  for task in c.TASK_LIST:

    tasks = "{directoryPath}/{project_name}/{project_name}_dataset_{task}.csv".format(directoryPath=directoryPath, project_name=project, task = task)
    
    # BEGIN Core Contributors
    df = pd.read_csv(tasks)
    df[c.T_LINE] = df[c.T_LINE].shift()
    df.fillna(0, inplace=True)

    corr_module_nt_cc, _ = pearsonr(df[c.MODULE_CC], df[c.NT_CC])
    corr_module_no_cc, _ = pearsonr(df[c.MODULE_CC], df[c.NO_CC])
    corr_module_t_cc, _ = pearsonr(df[c.MODULE_CC], df[c.T_CC])
    corr_module_nt_ec, _ = pearsonr(df[c.MODULE_CC], df[c.NT_CC])
    corr_module_no_ec, _ = pearsonr(df[c.MODULE_CC], df[c.NO_CC])
    corr_module_t_ec, _ = pearsonr(df[c.MODULE_CC], df[c.T_CC])
    corr_module_t_line, _ = pearsonr(df[c.MODULE_CC], df[c.T_LINE])


    correlationsFile.write('{0}:{1} - Pearsons correlation:\n'.format(project, task))
    correlationsFile.write('E_Module => NT_CC: {0}\n'.format(corr_module_nt_cc))
    correlationsFile.write('E_Module => NO_CC: {0}\n'.format(corr_module_no_cc))
    correlationsFile.write('E_Module => T_CC: {0}\n'.format(corr_module_t_cc))
    correlationsFile.write('E_Module => NT_EC: {0}\n'.format(corr_module_nt_ec))
    correlationsFile.write('E_Module => NO_CC: {0}\n'.format(corr_module_no_ec))
    correlationsFile.write('E_Module => T_EC: {0}\n'.format(corr_module_t_ec))
    correlationsFile.write('E_Module => T_Line: {0}\n\n'.format(corr_module_t_line))

    corr_line_nt_cc, _ = pearsonr(df[c.LINE_CC], df[c.NT_CC])
    corr_line_no_cc, _ = pearsonr(df[c.LINE_CC], df[c.NO_CC])
    corr_line_t_cc, _ = pearsonr(df[c.LINE_CC], df[c.T_CC])
    corr_line_nt_ec, _ = pearsonr(df[c.LINE_CC], df[c.NT_CC])
    corr_line_no_ec, _ = pearsonr(df[c.LINE_CC], df[c.NO_CC])
    corr_line_t_ec, _ = pearsonr(df[c.LINE_CC], df[c.T_CC])
    corr_line_t_line, _ = pearsonr(df[c.LINE_CC], df[c.T_LINE])

    correlationsFile.write('{0}:{1} - Pearsons correlation:\n'.format(project, task))
    correlationsFile.write('E_Line => NT_CC: {0}\n'.format(corr_line_nt_cc))
    correlationsFile.write('E_Line => NO_CC: {0}\n'.format(corr_line_no_cc))
    correlationsFile.write('E_Line => T_CC: {0}\n'.format(corr_line_t_cc))
    correlationsFile.write('E_Line => NT_EC: {0}\n'.format(corr_line_nt_ec))
    correlationsFile.write('E_Line => NO_CC: {0}\n'.format(corr_line_no_ec))
    correlationsFile.write('E_Line => T_EC: {0}\n'.format(corr_line_t_ec))
    correlationsFile.write('E_Line => T_Line: {0}\n\n'.format(corr_line_t_line))

    corr_module_nt_cc, _ = spearmanr(df[c.MODULE_CC], df[c.NT_CC])
    corr_module_no_cc, _ = spearmanr(df[c.MODULE_CC], df[c.NO_CC])
    corr_module_t_cc, _ = spearmanr(df[c.MODULE_CC], df[c.T_CC])
    corr_module_nt_ec, _ = spearmanr(df[c.MODULE_CC], df[c.NT_CC])
    corr_module_no_ec, _ = spearmanr(df[c.MODULE_CC], df[c.NO_CC])
    corr_module_t_ec, _ = spearmanr(df[c.MODULE_CC], df[c.T_CC])
    corr_module_t_line, _ = spearmanr(df[c.MODULE_CC], df[c.T_LINE])


    correlationsFile.write('{0}:{1} - Spearman correlation:\n'.format(project, task))
    correlationsFile.write('E_Module => NT_CC: {0}\n'.format(corr_module_nt_cc))
    correlationsFile.write('E_Module => NO_CC: {0}\n'.format(corr_module_no_cc))
    correlationsFile.write('E_Module => T_CC: {0}\n'.format(corr_module_t_cc))
    correlationsFile.write('E_Module => NT_EC: {0}\n'.format(corr_module_nt_ec))
    correlationsFile.write('E_Module => NO_CC: {0}\n'.format(corr_module_no_ec))
    correlationsFile.write('E_Module => T_EC: {0}\n'.format(corr_module_t_ec))
    correlationsFile.write('E_Module => T_Line: {0}\n\n'.format(corr_module_t_line))

    corr_line_nt_cc, _ = spearmanr(df[c.LINE_CC], df[c.NT_CC])
    corr_line_no_cc, _ = spearmanr(df[c.LINE_CC], df[c.NO_CC])
    corr_line_t_cc, _ = spearmanr(df[c.LINE_CC], df[c.T_CC])
    corr_line_nt_ec, _ = spearmanr(df[c.LINE_CC], df[c.NT_CC])
    corr_line_no_ec, _ = spearmanr(df[c.LINE_CC], df[c.NO_CC])
    corr_line_t_ec, _ = spearmanr(df[c.LINE_CC], df[c.T_CC])
    corr_line_t_line, _ = spearmanr(df[c.LINE_CC], df[c.T_LINE])

    correlationsFile.write('{0}:{1} - Spearman correlation:\n'.format(project, task))
    correlationsFile.write('E_Line => NT_CC: {0}\n'.format(corr_line_nt_cc))
    correlationsFile.write('E_Line => NO_CC: {0}\n'.format(corr_line_no_cc))
    correlationsFile.write('E_Line => T_CC: {0}\n'.format(corr_line_t_cc))
    correlationsFile.write('E_Line => NT_EC: {0}\n'.format(corr_line_nt_ec))
    correlationsFile.write('E_Line => NO_CC: {0}\n'.format(corr_line_no_ec))
    correlationsFile.write('E_Line => T_EC: {0}\n'.format(corr_line_t_ec))
    correlationsFile.write('E_Line => T_Line: {0}\n\n'.format(corr_line_t_line))

  correlationsFile.close()

    



