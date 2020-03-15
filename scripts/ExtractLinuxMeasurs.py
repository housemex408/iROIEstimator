import subprocess

tags = open('tags.txt', 'r')
data_analysis = open("./output/data_analysis.csv", "w")
header = 'Version, NC, NO, E_Module, E_Line'
data_analysis.write(header)
fromVersion = tags.readline()
toVersion = tags.readline()


while True:
    toVersion = toVersion.strip()
    fromVersion = fromVersion.strip()

    versionRange = "'{fromV}'..'{toV}'".format(toV=toVersion, fromV=fromVersion)
    print(versionRange)

    changeLog = '{filename}_changelog.txt'.format(filename=toVersion)

    # Get the change log for the current versionRange
    getChangeLog = 'git log --pretty="%h - %s (%an)" {range} > ../scripts/output/{outputFile}'.format(range=versionRange, outputFile=changeLog)

    #print(getChangeLog)
    process = subprocess.run([getChangeLog], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd='../linux/')

 
    # Get corrective tasks
    getNC = 'grep -Ec "fix|bug|defect" {outputFile}'.format(outputFile=changeLog)

    #print(getCorrectiveTasks)

    process = subprocess.run([getNC], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd='./output/')
    NC = process.stdout.strip()
    msg = 'NC: {NC}'.format(NC=NC)
    print(msg)

    # Get other tasks
    getNO = 'grep -Ecv "fix|bug|defect" {outputFile}'.format(outputFile=changeLog)

    #print(getOtherTasks)

    process = subprocess.run([getNO], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd='./output/')
    NO = process.stdout.strip()
    msg = 'NO: {NO}'.format(NO=NO)
    print(msg)

    # Get files modified / added / deleted
    getEModule = "git diff --shortstat " + versionRange + " | awk '{print $1}'"

    #print(getEModule)

    process = subprocess.run([getEModule], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd='../linux/')
    E_Module = process.stdout.strip()
    msg = 'E_Module: {E_Module}'.format(E_Module=E_Module)
    print(msg)

    # Get LOC modified / added / deleted
    getELine = "git diff --shortstat " + versionRange + " | awk '{print $4 + $6}'"

    #print(getELine)

    process = subprocess.run([getELine], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd='../linux/')
    E_Line = process.stdout.strip()
    msg = 'E_Line: {E_Line}'.format(E_Line=E_Line)
    print(msg)

    # Get total files up to this version

    # Get total LOC up to this version

    # Create a line item { toVersion, NC, NO, E_Module, E_Line, T_Module, T_Line }
    
    template = '\n{toVersion}, {NC}, {NO}, {E_Module}, {E_Line}'
    line = template.format(toVersion=toVersion, NC=NC, NO=NO, E_Module=E_Module, E_Line=E_Line)
    data_analysis.write(line)

    fromVersion = toVersion

    toVersion = tags.readline()
    #print(toVersion.strip())

    if not toVersion:
        break

tags.close()
data_analysis.close()

