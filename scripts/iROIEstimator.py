import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(__file__))

print(__file__)
import Utilities as utils
import Constants as c

class iROIEstimator:
    cwd = "scripts/exports"

    def __init__(self, project):
        self.project_name = project.split('/')[1];
        self.file_template = "{cwd}/{project_name}/{project_name}_dataset_{task}.csv"

    def predictEffort(self):
        for task in c.TASK_LIST:
            tasks = self.file_template.format(cwd=self.cwd, project_name=self.project_name, task = task)
            df = pd.read_csv(tasks)
            print(df.head())

    # def forecastEffort():

    # def calculateROI():

    # def printLine():

angular = iROIEstimator("angular/angular")
angular.predictEffort()