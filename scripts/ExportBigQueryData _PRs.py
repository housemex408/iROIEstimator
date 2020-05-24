import os
import google.cloud
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(__file__))
import Utilities as utils
import Constants as c
from google.cloud import bigquery

def create_directory(name):
  try:
    os.mkdir(name)
  except OSError as error:
      print(error)

header = [
  c.PROJECT, c.VERSION, c.DATE, c.TASK, c.T_LINE, c.T_MODULE, c.NT, 
  c.T_COMMITS, c.MODULE, c.LINE, c.CONTRIBUTIONS, c.CONTRIBUTORS,
  c.COMMENTS, c.COMMENTERS
]

for project_name in ["angular/angular"]:

  print(project_name)

  dir_name = project_name.split('/')[1]
  outputDirectoryPath = "scripts/exports/{directory}".format(directory=dir_name)
  create_directory(outputDirectoryPath)
  versionsFile = "{outputDirectoryPath}/{filename}_dataset_BUG.csv".format(outputDirectoryPath = outputDirectoryPath, filename = dir_name)
  versions = pd.read_csv(versionsFile)

  for index, row in versions.iterrows():

    version = row[c.VERSION]
    print(version)

    for task_name in c.TASK_LIST:
    # for task_name in ["TESTING"]:

      outputFileName = "{outputDirectoryPath}/{filename}_PRs_{task}.csv".format(
          outputDirectoryPath=outputDirectoryPath,
          filename=dir_name,
          task=task_name
      )

      df = pd.DataFrame(columns = header)

      if not os.path.isfile(outputFileName):
          df.to_csv(outputFileName, index=False)

      print(task_name)

      client = bigquery.Client()
      sql = """
          DECLARE p_next STRING;
          DECLARE t_next STRING;
          DECLARE v_next STRING;
            
          SET p_next = @project_name;
          SET t_next = @task_name;
          SET v_next = @version;

          CREATE TABLE IF NOT EXISTS `praxis.PR`
          (Project_id INT64, Project String, Version String, Task String, pr_id INT64, pullreq_id INT64)
          OPTIONS(
            expiration_timestamp=TIMESTAMP "2021-01-01 00:00:00 UTC",
            description="Export PR tasks.  Expires in 2021"
          );

          INSERT INTO `praxis.PR`
          SELECT DISTINCT p.id as Project_id, tt.Key as Project, tt.Version, tt.task, pr.id as pr_id, pr.pullreq_id
          FROM `praxis.task_types` as tt
          INNER JOIN `praxis.ghtorrent_projects_top_30` as p
          ON tt.Key = SUBSTR(p.url, 30) AND tt.Key = p_next AND tt.Version = v_next  
          INNER JOIN `praxis.ghtorrent_commits` as c
          ON c.project_id = p.id AND c.sha = tt.sha
          INNER JOIN `praxis.ghtorrent_pull_requests` as pr
          ON (pr.base_repo_id = p.id AND pr.base_commit_id = c.id) OR (pr.head_repo_id = p.id AND pr.head_commit_id = c.id)
          INNER JOIN `praxis.ghtorrent_pull_request_history` as ph
          ON ph.pull_request_id = pr.id
          WHERE tt.Key = p_next 
          AND tt.Version = v_next
          AND tt.Task = t_next
          AND ph.action in ('closed')
          ORDER BY pullreq_id;

          SELECT pr.Project, pr.Version, pr.Date, pr.Task, pr.T_Line, pr.T_Module, count(DISTINCT pr.pr_id) as N_T,
          sum(pr.T_Commits) as T_Commits, sum(pr.Module) as Module, sum(pr.Line) as Line, sum(pr.Contributions) as Contributions, 
          sum(pr.Contributors) as Contributors, sum(c.Comments) as Comments, sum(c.Commenters) as Commenters
          FROM
          (
            SELECT pr.pr_id, pr.Project, pr.Version, vm.Release_Date as Date, tt.Task, vm.T_Line, vm.T_Module,
            count(prc.commit_id) as T_Commits, SUM(rc.E_Module) as Module, SUM(rc.E_Line) as Line, 
            count(c.commiter_id) as Contributions, count(DISTINCT c.commiter_id) as Contributors
            FROM `praxis.ghtorrent_projects_top_30` as p
            INNER JOIN `praxis.PR` as pr
            ON SUBSTR(p.url, 30) = pr.Project
            INNER JOIN `praxis.ghtorrent_commits` as c
            ON c.project_id = p.id
            LEFT JOIN `praxis.ghtorrent_pull_request_commits` prc
            ON prc.pull_request_id = pr.pr_id AND prc.commit_id = c.id
            LEFT JOIN `praxis.repo_repository_commits` as rc
            ON rc.Key = pr.Project AND rc.Version = pr.Version AND rc.SHA = c.SHA
            LEFT JOIN `praxis.task_types` as tt
            ON tt.Key = rc.Key AND tt.Version = rc.Version AND tt.sha = c.sha AND tt.task = pr.task
            INNER JOIN `praxis.repo_version_metrics` as vm
            ON vm.Key = rc.Key AND vm.Version = rc.Version
            WHERE tt.task = t_next
            AND tt.Key = p_next
            AND tt.Version = v_next
            GROUP BY pr.pr_id, pr.Project, pr.Version, vm.Release_Date, tt.Task, vm.T_Line, vm.T_Module
          ) AS pr
          LEFT JOIN (
            SELECT pull_request_id, count(comment_id) as Comments, count(DISTINCT user_id) as Commenters
              FROM
              (
                SELECT prc.comment_id, prc.user_id, prc.pull_request_id
                FROM `praxis.ghtorrent_pull_request_comments` as prc
                WHERE prc.pull_request_id in (SELECT pr_id FROM `praxis.PR`)
                UNION ALL
                SELECT ic.comment_id, ic.user_id, i.pull_request_id
                FROM `praxis.ghtorrent_issue_comments` ic, `praxis.ghtorrent_issues` as i
                WHERE i.id = ic.issue_id and i.pull_request_id in (SELECT pr_id FROM `praxis.PR`)
              )
              GROUP BY pull_request_id
          ) as c
          ON pr.pr_id = c.pull_request_id
          GROUP BY pr.Project, pr.Version, pr.Date, pr.Task, pr.T_Line, pr.T_Module;
      """
      query_config = bigquery.QueryJobConfig(
          query_parameters=[
              bigquery.ScalarQueryParameter("project_name", "STRING", project_name),
              bigquery.ScalarQueryParameter("task_name", "STRING", task_name),
              bigquery.ScalarQueryParameter("version", "STRING", version),
          ]
      )

      df = client.query(sql, job_config=query_config).to_dataframe()

      if len(df) == 0:
        data = {}
        data[c.PROJECT] = row[c.PROJECT]
        data[c.VERSION] = row[c.VERSION]
        data[c.DATE] = row[c.DATE]
        data[c.TASK] = task_name
        data[c.T_LINE] = row[c.T_LINE]
        data[c.T_MODULE] = row[c.T_MODULE]

        data[c.NT] = np.nan
        data[c.T_COMMITS] = np.nan
        data[c.MODULE] = np.nan
        data[c.LINE] = np.nan
        data[c.CONTRIBUTIONS] = np.nan
        data[c.CONTRIBUTORS] = np.nan
        data[c.COMMENTS] = np.nan
        data[c.COMMENTERS] = np.nan

        df = pd.DataFrame([data], columns = header)
        # df.loc[len(df)] = data
        df.to_csv(outputFileName, header=False, mode = 'a', index=False)
      else:
        df.to_csv(outputFileName, header=False, mode = 'a', index=False)
