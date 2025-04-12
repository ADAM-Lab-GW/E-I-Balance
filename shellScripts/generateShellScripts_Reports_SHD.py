
fileNum=152000
numJobsPerHour = None # adds a delay to submit jobs
                        # remove the delay by using None
baseFileName = './shellScripts/genReport_SHD_'
commandsPerJob = 4
commandLength = 55 # minutes
jobLength = commandsPerJob * commandLength

# automatically determine the smallest
#  node for the job
node = 'defq'
if jobLength < 30:
    node = 'nano'
elif jobLength < 240:
    node = 'tiny'
elif jobLength < 1440:
    node = 'short'

# create the string for the job time length
jobTime = '14-00:00:00'
if jobLength < 60:
    jobTime = str(jobLength)+':00'
elif jobLength < 1440:
    hours = int(jobLength/60)
    minutes = int(jobLength%60)
    jobTime = f'{hours:02d}:{minutes:02d}:00'
else:
    days = int(jobLength/1440)
    hours = int((jobLength-(days*1440))/60)
    minutes = int(jobLength-(days*1440)-(hours*60))
    jobTime = f'{days:02d}-{hours:02d}:{minutes:02d}:00'

print('GENERATING SHELL SCRIPTS')
print('fileNums: '+str(fileNum))
print('numJobsPerHour: '+str(numJobsPerHour))
print('baseFileName: '+baseFileName)
print('commandsPerJob: '+str(commandsPerJob))
print('commandLength: '+str(commandLength))
print('jobLength: '+str(jobLength))
print('node: '+node)
print('jobTime: '+jobTime)


def generateCommand(reportFile='SHD_50_t0_e-1.html',
        dataSet='SHD_50',
        trialRange=[0,1],
        activity=False,
        distance=False,
        epoch=-1,
        objFolder='./data/SHD_50_t0'):
    command = 'python3.7 scripts/generateReportSHD.py'
    command += ' -r ' + reportFile
    command += ' -s ' + dataSet
    command += ' -t ' + str(trialRange[0]) + ' ' + str(trialRange[1])
    if activity:
        command += ' -a '
    if distance:
        command += ' -d '
    command += ' -e ' + str(epoch)
    command += ' -o ' + objFolder
    return command

def genArgList(dataSet, trial, activity, distance, epoch):
    reportFile = dataSet+'_'+str(trial)+'_e'+str(epoch)+'.html'
    trials = [trial, trial+1]
    objFolder = './data/'+dataSet+'/obj'
    return [reportFile, dataSet, trials, activity, distance, epoch, objFolder]

args = []
for e in ['050','080','095','100']:
    for t in range(100, 244):
        for i in [-1,199]:
            args.append(genArgList('SHD_'+e, t, True, True, i))

# use this to find which reports to generate baseed on a simTags list file
#with open('./shellScripts/simtagsToBeAnalyzed.txt','r') as f:
#    for line in f.readlines():
#        if('#' in line): continue
#
#        for i in range(-1, 30):
#            args.append(genArgListSimTag(line.replace('\n',''), True, True, i))

shellFiles=[]
comIndex = 1
for [reportFile, dataSet, trials, activity, distance, epoch, objFolder] in args:
    if comIndex == 1:
        shellScriptFile = baseFileName+str(fileNum)+'.sh'
        f = open(shellScriptFile, 'w')
        f.write('#!/bin/bash \n')
        f.write('\n')
        f.write('#SBATCH -o LOG%j.out\n')
        f.write('#SBATCH -e LOG%j.err\n')
        f.write('#SBATCH -p '+node+'\n')
        f.write('#SBATCH -N 1\n')
        f.write('#SBATCH -D /lustre/groups/adamgrp/repos/surrogate-learning\n')
        f.write('#SBATCH -J '+dataSet+'_'+str(trials[0])+'_'+str(epoch)+'\n')
        f.write('#SBATCH --export=NONE\n')
        f.write('#SBATCH -t '+jobTime+'\n')
        f.write('#SBATCH --nice=100\n')
        f.write('\n')
        f.write('module load python3/3.7.2\n')
        

    command = generateCommand(reportFile=reportFile,
            dataSet=dataSet,
            trialRange=trials,
            activity=activity,
            distance=distance,
            epoch=epoch,
            objFolder=objFolder)
    f.write(command+'\n')
    
    if comIndex == commandsPerJob:
        f.close()
        shellFiles.append(shellScriptFile)
        fileNum += 1
        comIndex = 0
    
    comIndex += 1

f = open('runMultipleGenerateReports.sh', 'w')
f.write('#!/bin/bash\n')
f.write('\n')
i = 0
for shellFile in shellFiles:
    f.write('sbatch ')
    if numJobsPerHour != None and i>numJobsPerHour:
        jobDelay = int(i/numJobsPerHour) # delay jobs so that
                                        # we only run a certain number per hour
        f.write('--begin=now+'+str(jobDelay)+'hour ')
    f.write(shellFile+'\n')
    i+=1
f.close()

print('Total scripts generated: '+str(i)+'\n\n')
