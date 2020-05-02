import subprocess
import os
from subprocess import Popen
from itertools import islice

repos = open("./scripts/less_popular_repos.csv", 'r')
projects = []

for repo in repos:
    projects.append("git clone {repo}".format(repo=repo))

max_workers = 5
processes = (Popen(cmd, shell=True, cwd="/Users/alvaradojo/Documents/Github") for cmd in projects)
running_processes = list(islice(processes, max_workers))  # start new processes
while running_processes:
    for i, process in enumerate(running_processes):
        if process.poll() is not None:  # the process has finished
            running_processes[i] = next(processes, None)  # start new process
            if running_processes[i] is None: # no new processes
                del running_processes[i]
                break
