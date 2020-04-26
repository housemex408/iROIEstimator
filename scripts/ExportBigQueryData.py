import os
import google.cloud
from google.cloud import bigquery

project_list = [
  "angular/angular"
  , "nodejs/node"
  , "openstack/neutron"
  , "vuejs/vue"
  ,"home-assistant/home-assistant"
  ,"tensorflow/tensorflow"
  ,"moby/moby"
  ,"gitlabhq/gitlabhq"
  ,"dotnet/orleans"
  ,"dotnet/roslyn"
  ,"ansible/ansible"
  ,"cloudfoundry/cli"
  ,"openstack/nova"
  ,"angular/angular.js"
  ,"auth0/lock"
  ,"kubernetes/kubernetes"
  ,"apache/mesos"
  ,"odin-lang/Odin"
  ,"NixOS/nixpkgs"
  ,"facebook/react"
  ,"Homebrew/brew"
  ,"openstack/cinder"
  ,"elastic/elasticsearch"
  ,"torvalds/linux"
  ,"cloudfoundry/cf-deployment"
  ,"OfficeDev/office-js"
]
task_list = ["BUG", "DOCS", "REFACTOR", "TESTING", "FEATURE", "UPGRADE", "RELEASE", "SUPPORT", "OTHER"]

def create_directory(name):
  try:
    os.mkdir(name)
  except OSError as error:
      print(error)

for project_name in project_list:

  dir_name = repo = project_name.split('/')[1]
  outputDirectoryPath = "scripts/exports/{directory}".format(directory=dir_name) 
  create_directory(outputDirectoryPath)

  for task_name in task_list:

    outputFileName = "{outputDirectoryPath}/{filename}_dataset_{task}.csv".format(
        outputDirectoryPath=outputDirectoryPath, 
        filename=dir_name,
        task=task_name
    )

    client = bigquery.Client()
    sql = """
        SELECT vm.Key as Project, vm.Version, vm.Release_Date as Date, cc.Task, cc.T_Module, cc.T_Line, 
        cc.N_T as NT_CC, cc.N_O as NO_CC, cc.Module as Module_CC, cc.Line as Line_CC, cc.Contributors as T_CC,
        ec.N_T as NT_EC, ec.N_O as NO_EC, ec.Module as Module_EC, ec.Line as Line_EC, ec.Contributors as T_EC
        FROM `praxis.repo_version_metrics` as vm
        LEFT JOIN 
        ( 
        SELECT * 
        FROM `praxis.cc_commit_tasks`
        WHERE Project = @project_name
        AND Task = @task_name 
        ) AS cc
        ON cc.Project = vm.Key AND cc.Version = vm.Version
        LEFT JOIN
        (
        SELECT * 
        FROM `praxis.ec_commit_tasks`
        WHERE Project = @project_name
        AND Task = @task_name 
        ) AS ec
        ON ec.Project = vm.Key AND ec.Version = vm.Version
        WHERE vm.Key = @project_name

        ORDER BY vm.Key, vm.Release_Date, vm.T_Line, IFNULL(cc.Task, 'Z') ASC, vm.Version DESC
    """
    query_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("project_name", "STRING", project_name),
            bigquery.ScalarQueryParameter("task_name", "STRING", task_name),
        ]
    )

    df = client.query(sql, job_config=query_config).to_dataframe()
    df.to_csv(outputFileName)