import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import shapiro
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
from Effort import Effort
import concurrent.futures
import fcntl
logger = utils.get_logger()
os.environ["NUMEXPR_MAX_THREADS"] = "12"

class iROIEstimator:
    # input = "../../exports"
    input = "scripts/exports"
    output = "scripts/notebook/results"
    TASK_LIST = c.TASK_LIST
    # TASK_LIST = ["BUG"]

    results_header = [
      c.DATE, c.PROJECT, c.MODEL, c.TASK, c.NT, c.NO, c.T_CONTRIBUTORS,
      c.AVG_MODULE_CONTRIBS, c.HOURS_DIFF, c.CONTRIB_DIFF, c.BILLED_HOURS, c.COST,
      c.T_LINE_P, c.OBSERVED, c.PREDICTED, c.DIFFERENCE, c.PERCENT_ERROR
    ]
    performance_measures_header = [c.PROJECT, c.MODEL, c.TASK, c.R_SQUARED, c.R_SQUARED_ADJ, c.MAE, c.MSE, c.RMSE, c.PRED_25, c.PRED_50, c.T_RECORDS]
    roi_header = [c.PROJECT, c.MODEL, c.AMOUNT_INVESTED, c.AMOUNT_RETURNED, c.INVESTMENT_GAIN, c.ROI, c.ANNUALIZED_ROI]

    def __init__(self, project, model=c.LINE, prediction_years=3):
        self.project_name = project.split('/')[1]
        self.file_template = "{cwd}/{project_name}/{project_name}_dataset_{task}.csv"
        self.model = model
        self.line_cc = None
        self.line_ec = None
        self.module_cc = None
        self.module_ec = None
        self.prediction_years = prediction_years
        self.predicton_months = self.prediction_years * 12
        self.task_forecasted_effort = {}
        self.amount_invested = 0
        self.amount_returned = 0
        self.investment_gain = 0
        self.roi = 0
        self.annualized_roi = 0
        self.init_output_files(model)


    def init_output_files(self, model):
        self.results_file = "{directory}/prediction_results_{model}.csv".format(directory=self.output, model=model)
        self.performance_measures_file = "{directory}/performance_measures_{model}.csv".format(directory=self.output, model=model)
        self.roi_measures_file = "{directory}/roi_measures_{model}.csv".format(directory=self.output, model=model)

        self.results = pd.DataFrame(columns = self.results_header)
        self.performance_measures = pd.DataFrame(columns = self.performance_measures_header)
        self.roi_measures = pd.DataFrame(columns = self.roi_header)

        if not os.path.isfile(self.results_file):
          with open(self.results_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            self.results.to_csv(f, mode = 'a', index=False)
            fcntl.flock(f, fcntl.LOCK_UN)

        if not os.path.isfile(self.performance_measures_file):
          with open(self.performance_measures_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            self.performance_measures.to_csv(f, mode = 'a', index=False)
            fcntl.flock(f, fcntl.LOCK_UN)

        if not os.path.isfile(self.roi_measures_file):
          with open(self.roi_measures_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            self.roi_measures.to_csv(f, mode = 'a', index=False)
            fcntl.flock(f, fcntl.LOCK_UN)

    def execute(self):
        # for task in ["BUG"]:
        for task in self.TASK_LIST:
            tasks = self.file_template.format(cwd=self.input, project_name=self.project_name, task = task)
            df = pd.read_csv(tasks)

            # df = df.dropna(subset=[c.T_MODULE])
            df = df.dropna(subset=[c.DATE])
            # df = df.dropna(subset=[c.TASK])
            df[c.DATE] = pd.to_datetime(df[c.DATE])
            # df = df.set_index(c.DATE)
            # df.index.name = c.DATE
            # df.index = df.index.strftime('%Y-%m-%d')

            if df.isna().values.any():
                df.fillna(0, inplace=True)

            t_records = len(df)

            # Edge case when < 2 tasks detected
            if t_records < 2:
                break

            results, performance_measures = self.predict_effort(task, df)

            self.results = pd.concat([self.results, results])
            self.performance_measures = pd.concat([self.performance_measures, performance_measures])

            self.forecast_effort(df, task)
        self.display_forecast(self.prediction_years)
        self.save_results_performance_measures()
        self.calculate_results()
        self.save_results_roi_measures()

    def get_independent_variables(self):
        CC = c.LINE_CC
        EC = c.LINE_EC

        if self.model == c.MODULE:
            CC = c.MODULE_CC
            EC = c.MODULE_EC

        return CC, EC

    def predict_effort(self, task, df):

        CC, EC = self.get_independent_variables()

        self.module_cc = Effort(self.project_name, self.model, CC, task, df)
        module_cc_results = self.module_cc.predict_effort()
        logger.info("{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, CC, module_cc_results.size))
        self.module_cc.calculate_perf_measurements()
        module_cc_output = self.module_cc.create_output_df()

        self.module_ec = Effort(self.project_name, self.model, EC, task, df)
        module_ec_results = self.module_ec.predict_effort()
        logger.info("{0} - {1} - {2} prediction count: {3}".format(self.project_name, task, EC, module_ec_results.size))
        self.module_ec.calculate_perf_measurements()
        module_ec_output = self.module_ec.create_output_df()

        results_df = pd.concat([module_cc_results, module_ec_results])
        output_df = pd.concat([module_cc_output, module_ec_output])

        return results_df, output_df

    def forecast_effort(self, df, task):
        self.module_cc.forecast_module_effort(self.predicton_months)
        self.module_ec.forecast_module_effort(self.predicton_months)

        self.task_forecasted_effort[task] = {
            c.COST_CC: self.module_cc.calculate_total_effort(self.prediction_years),
            c.COST_EC: self.module_ec.calculate_total_effort(self.prediction_years)
        }

    def display_forecast(self, predictionYears):
        for key in self.task_forecasted_effort:
            logger.info("{0} - {1} Forecasted Effort: \n".format(self.project_name, key))
            logger.info(self.task_forecasted_effort[key][c.COST_CC].T)
            logger.info(self.task_forecasted_effort[key][c.COST_EC].T)

            self.results.sort_values(by=[c.DATE, c.PROJECT, c.MODEL, c.TASK], ascending = True, inplace=True)
            self.performance_measures.sort_values(by=[c.PROJECT, c.MODEL, c.TASK], inplace=True)
            logger.info("\n {0}".format(self.results))
            logger.info("\n {0}".format(self.performance_measures))

    def save_results_performance_measures(self):
        with open(self.results_file, "a") as f:
          fcntl.flock(f, fcntl.LOCK_EX)
          self.results.to_csv(f, header=False, mode = 'a', index=False)
          fcntl.flock(f, fcntl.LOCK_UN)

        with open(self.performance_measures_file, "a") as f:
          fcntl.flock(f, fcntl.LOCK_EX)
          self.performance_measures.to_csv(f, header=False, mode = 'a', index=False)
          fcntl.flock(f, fcntl.LOCK_UN)

    def calculate_investment_gain(self):
        self.investment_gain = round(self.amount_returned - self.amount_invested, 2)
        return self.investment_gain

    def calculate_ROI(self):
        self.roi = round(((self.investment_gain + self.amount_invested) / self.amount_invested) - 1, 3)
        return self.roi

    def calculate_annualized_ROI(self):
        self.annualized_roi = round(pow(1 + self.roi, 1 / self.prediction_years) - 1, 3)
        return self.annualized_roi

    def calculate_savings(self, effort_cc, cost_cc, effort_ec):
        cost_ec = (cost_cc / effort_cc) * effort_ec
        return cost_ec


    def calculate_results(self):
        effort_cc = 0.0
        effort_ec = 0.0
        self.amount_invested = 0
        self.amount_returned = 0
        self.investment_gain = 0
        self.roi = 0
        self.annualized_roi = 0

        cost_cc = 0
        cost_ec = 0
        cost_invested = 0
        cost_returned = 0

        for key in self.task_forecasted_effort:
            effort_cc = self.task_forecasted_effort[key][c.COST_CC].iloc[:,1].values.sum()
            effort_ec = self.task_forecasted_effort[key][c.COST_EC].iloc[:,1].values.sum()
            cost_cc = self.task_forecasted_effort[key][c.COST_CC].iloc[:,0].values.sum()
            cost_ec = self.calculate_savings(effort_cc, cost_cc, effort_ec)

            cost_invested = cost_invested + cost_cc
            # cost_returned = cost_returned + cost_ec
            self.amount_invested = self.amount_invested + effort_cc
            self.amount_returned = self.amount_returned + effort_ec

            logger.info("\n{0} - {1} CC Forecasted Effort: {2}".format(self.project_name, key, round(effort_cc), 2))
            logger.info("{0} - {1} CC Forecasted Costs: {2}".format(self.project_name, key, round(cost_cc), 2))
            logger.info("{0} - {1} EC Forecasted Effort Savings: {2}".format(self.project_name, key, round(effort_ec), 2))
            logger.info("{0} - {1} EC Forecasted Costs Savings: {2}".format(self.project_name, key, round(cost_ec), 2))

        cost_returned = self.calculate_savings(self.amount_invested, cost_invested, self.amount_returned)

        logger.info("\n{0} - Core Contributor Forecasted Effort Over {1} years: {2}".format(self.project_name, self.prediction_years, round(self.amount_invested, 2)))
        logger.info("{0} - Core Contributor Forecasted Costs Over {1} years: {2}".format(self.project_name, self.prediction_years, round(cost_invested, 2)))
        logger.info("{0} - External Contributor Forecasted Effort Savings Over {1} years: {2}".format(self.project_name, self.prediction_years, round(self.amount_returned, 2)))
        logger.info("{0} - External Contributor Forecasted Costs Savings Over {1} years: {2}".format(self.project_name, self.prediction_years, round(cost_returned, 2)))

        self.calculate_investment_gain()
        self.calculate_ROI()
        self.calculate_annualized_ROI()

        logger.info("{0} - Investment Gain: {1}".format(self.project_name, self.investment_gain))
        logger.info("{0} - ROI: {1}".format(self.project_name, self.roi))
        logger.info("{0} - Annualized ROI: {1}".format(self.project_name, self.annualized_roi))

    def save_results_roi_measures(self):
        roi_measures = pd.DataFrame({c.PROJECT: [self.project_name],
                      c.MODEL: [self.model],
                      c.AMOUNT_INVESTED: [self.amount_invested],
                      c.AMOUNT_RETURNED: [self.amount_returned],
                      c.INVESTMENT_GAIN: [self.investment_gain],
                      c.ROI: [self.roi],
                      c.ANNUALIZED_ROI: [self.annualized_roi]})

        self.roi_measures = pd.concat([self.roi_measures, roi_measures])

        with open(self.roi_measures_file, "a") as f:
          fcntl.flock(f, fcntl.LOCK_EX)
          self.roi_measures.to_csv(f, header=False, mode = 'a', index=False)
          fcntl.flock(f, fcntl.LOCK_UN)

# project_list = c.ALL_PROJECTS
project_list = ["angular/angular"]

# def execute_iROIEstimator(p, model):
#   try:
#     logger.debug("Project {0}".format(p))
#     estimator = iROIEstimator(p, model)
#     estimator.execute()
#   except Exception:
#     logger.error("Error:  {0}".format(p), exc_info=True)


# with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
#     {executor.submit(execute_iROIEstimator, project, c.LINE): project for project in project_list}

for p in project_list:
  try:
    logger.debug("Project {0}".format(p))
    estimator = iROIEstimator(p, c.MODULE)
    estimator.execute()
  except Exception:
    logger.error("Error:  {0}".format(p), exc_info=True)
    continue

