import os
import google.cloud
import sys
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
from google.cloud import bigquery

def create_directory(name):
  try:
    os.mkdir(name)
  except OSError as error:
      print(error)

for project_name in c.ALL_PROJECTS:

  dir_name = repo = project_name.split('/')[1]
  outputDirectoryPath = "scripts/exports/{directory}".format(directory=dir_name)
  create_directory(outputDirectoryPath)

  for task_name in c.TASK_LIST:

    outputFileName = "{outputDirectoryPath}/{filename}_dataset_{task}.csv".format(
        outputDirectoryPath=outputDirectoryPath,
        filename=dir_name,
        task=task_name
    )

    client = bigquery.Client()
    sql = """
        SELECT vm.Key as Project, vm.Version, vm.Release_Date as Date, "{0}", vm.T_Module, vm.T_Line, 
        cc.N_T as NT_CC, cc.N_O as NO_CC, cc.Module as Module_CC, cc.Line as Line_CC, cc.Contributors as T_CC,
        ec.N_T as NT_EC, ec.N_O as NO_EC, ec.Module as Module_EC, ec.Line as Line_EC, ec.Contributors as T_EC,
        uc.N_T as NT_UC, uc.N_O as NO_UC, uc.Module as Module_UC, uc.Line as Line_UC, uc.Contributors as T_UC
        FROM `praxis.repo_version_metrics` as vm
        LEFT JOIN
        (
        SELECT *
        FROM `praxis.ccc_commit_tasks`
        WHERE Project = @project_name
        AND Task = @task_name
        ) AS cc
        ON cc.Project = vm.Key AND cc.Version = vm.Version
        LEFT JOIN
        (
        SELECT *
        FROM `praxis.ecc_commit_tasks`
        WHERE Project = @project_name
        AND Task = @task_name
        ) AS ec
        ON ec.Project = vm.Key AND ec.Version = vm.Version
        LEFT JOIN
        (
        SELECT * 
        FROM `praxis.uk_commit_tasks`
        WHERE Project = @project_name
        AND Task = @task_name 
        ) AS uc
        ON uc.Project = vm.Key AND uc.Version = vm.Version
        WHERE vm.Key = @project_name

        ORDER BY vm.Key, vm.Release_Date, vm.T_Line, IFNULL(cc.Task, 'Z') ASC, vm.Version DESC
    """.format(task_name)
    query_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("project_name", "STRING", project_name),
            bigquery.ScalarQueryParameter("task_name", "STRING", task_name),
        ]
    )

    df = client.query(sql, job_config=query_config).to_dataframe()
    df.to_csv(outputFileName)
