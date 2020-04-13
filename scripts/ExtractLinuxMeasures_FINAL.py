import subprocess
import os

# get list of tags
repo = "angular"
key = "angular/angular"
workingDirectory = "../{repo}".format(repo = repo)
outputDirectory = "../iROIEstimatorMetrics/{repo}".format(repo = repo)
tempDirectory = "{outputDirectory}/temp".format(outputDirectory = outputDirectory)
versionsFile = "{outputDirectory}/{repo}_versions.csv".format(outputDirectory = outputDirectory, repo = repo)

def create_directory(name):
  try:  
    os.mkdir(name)  
  except OSError as error:  
      print(error) 

create_directory(outputDirectory)
create_directory(tempDirectory)

# TODO:  add key to each row
getTags = "git for-each-ref --format '%(refname:lstrip=2)%2C %(creatordate:short)' refs/tags  --sort=creatordate > {versionsFile}".format(versionsFile = versionsFile)
process = subprocess.run([getTags], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=workingDirectory)
msg = process.stderr.strip()
print(msg)

# TODO:  get all commits along with files/loc and write to a file

tags = open(versionsFile, 'r')
versionMetricsFile = "{outputDirectory}/{repo}_version_metrics.csv".format(outputDirectory = outputDirectory, repo = repo)
data_analysis = open(versionMetricsFile, "w")
header = 'Key,Version,NC,NO,E_Module,E_Line,T_Module,T_Line,Release Date'
data_analysis.write(header)

class Tag(object):
    def __init__(self, row):
      self.version = row[0].strip()
      self.release_date = row[1].strip()
      self.print_template = 'Version: {version}, Release Date: {release_date}'
    def print(self):
        msg = self.print_template.format(version=self.version, release_date=self.release_date)
        print(msg)

seperator = ","

fromTag = Tag(tags.readline().split(seperator))
toTag = Tag(tags.readline().split(seperator))

fromTag.print()
toTag.print()

while True:
    toVersion = toTag.version
    Release_Date = toTag.release_date
    fromVersion = fromTag.version

    versionRange = "'{fromV}'..'{toV}'".format(toV=toVersion, fromV=fromVersion)
    print(versionRange)

    changeLog = '{tempDirectory}/{repo}_{filename}_changelog.txt'.format(tempDirectory = tempDirectory, repo = repo, filename=toVersion)

    # Get the change log for the current versionRange
    getChangeLog = 'git log --pretty="%h - %s (%an)" {range} > {outputFile}'.format(range=versionRange, outputFile=changeLog)

    #print(getChangeLog)
    process = subprocess.run([getChangeLog], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=workingDirectory)

    # TODO: get commits between two tags

    # Get corrective tasks
    getNC = 'grep -Ec "fix|bug|defect" {outputFile}'.format(outputFile=changeLog)

    #print(getCorrectiveTasks)

    process = subprocess.run([getNC], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    NC = process.stdout.strip()
    msg = 'NC: {NC}'.format(NC=NC)
    print(msg)

    # Get other tasks
    getNO = 'grep -Ecv "fix|bug|defect" {outputFile}'.format(outputFile=changeLog)

    #print(getOtherTasks)

    process = subprocess.run([getNO], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    NO = process.stdout.strip()
    msg = 'NO: {NO}'.format(NO=NO)
    print(msg)

    # Get files modified / added / deleted
    getEModule = "git diff --shortstat " + versionRange + " | awk '{print $1}'"

    #print(getEModule)

    process = subprocess.run([getEModule], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=workingDirectory)
    E_Module = process.stdout.strip()
    msg = 'E_Module: {E_Module}'.format(E_Module=E_Module)
    print(msg)

    # Get LOC modified / added / deleted
    getELine = "git diff --shortstat " + versionRange + " | awk '{print $4 + $6}'"

    #print(getELine)

    process = subprocess.run([getELine], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=workingDirectory)
    E_Line = process.stdout.strip()
    msg = 'E_Line: {E_Line}'.format(E_Line=E_Line)
    print(msg)

    # Checkout tags/toVersion
    checkoutTag = "git checkout tags/" + toVersion + " --force"
    print(checkoutTag)
    process = subprocess.run([checkoutTag], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=workingDirectory)
    print(process.stdout.strip())

    # Get total files up to this version
    getTotalFiles = "git ls-files --exclude-standard | wc -l | awk '{print $1 }'"
    process = subprocess.run([getTotalFiles], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=workingDirectory)
    T_Module = process.stdout.strip()
    msg = 'T_Module: {T_Module}'.format(T_Module=T_Module)
    print(msg)

    # Get total LOC up to this version
    getTotalLOC = "git grep ^ | wc -l | awk '{print $1 }'"
    process = subprocess.run([getTotalLOC], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=workingDirectory)
    T_Line = process.stdout.strip()
    msg = 'T_Line: {T_Line}'.format(T_Line=T_Line)
    print(msg)

    # Create a line item { toVersion, NC, NO, E_Module, E_Line, T_Module, T_Line }
    template = '\n{key},{toVersion},{NC},{NO},{E_Module},{E_Line},{T_Module},{T_Line},{Release_Date}'
    line = template.format(key=key,toVersion=toVersion, NC=NC, NO=NO, E_Module=E_Module, E_Line=E_Line, T_Module=T_Module, T_Line=T_Line, Release_Date=Release_Date)
    data_analysis.write(line)
    print(line)

    fromTag = toTag
    toTag = Tag(tags.readline().split(seperator))

    if not toVersion:
        break

tags.close()
data_analysis.close()

