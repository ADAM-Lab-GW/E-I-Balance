
numJobsPerHour = 101
baseFileName = './shellScripts/genReport_'

def generateCommand(reportFile='sparse050_10class_147_t7_e29.html',
        objPath='./data/obj',
        baseSimTag='sparse050_10class',
        trialRange=[147,148],
        activity=False,
        distance=False,
        epoch=29):
    command = 'python3.7 scripts/generateReport.py'
    command += ' -r ' + reportFile
    command += ' -s ' + baseSimTag
    command += ' -t ' + str(trialRange[0]) + ' ' + str(trialRange[1])
    if activity:
        command += ' -a '
    if distance:
        command += ' -d '
    command += ' -e ' + str(epoch)
    command += ' -o ' + objPath
    return command

def genArgList(dataSet, trial, activity, distance, epoch):
    trialSub = (trial-100)%8
    reportFile = dataSet+'_'+str(trial)+'_t'+str(trialSub)+'_e'+str(epoch)+'.html'
    trials = [trial, trial+1]
    return [reportFile, dataSet, trials, activity, distance, epoch]
    

args = []
for e in range(-1,200):
    args.append(['SHD_W1_Vary_9_'+str(e)+'.html','../data/SHD_W1_Vary_9','SHD_W1_Vary',[9,10],True,True,e])


fileNum=1000
shellFiles=[]
for [reportFile, objPath, baseSimTag, trials, activity, distance, epoch] in args:
    command = generateCommand(reportFile=reportFile,
        objPath=objPath,
        baseSimTag=baseSimTag,
        trialRange=trials,
        activity=activity,
        distance=distance,
        epoch=epoch)

    shellScriptFile = baseFileName+str(fileNum)+'.sh'
    f = open(shellScriptFile, 'w')
    f.write('#!/bin/bash \n')
    f.write('\n')
    f.write('#SBATCH -o LOG%j.out\n')
    f.write('#SBATCH -e LOG%j.out\n')
    f.write('#SBATCH -p nano\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH -D /lustre/groups/adamgrp/repos/surrogate-learning\n')
    f.write('#SBATCH -J Surr_training\n')
    f.write('#SBATCH --export=NONE\n')
    f.write('#SBATCH -t 29:59\n')
    f.write('#SBATCH --nice=100\n')
    f.write('\n')
    f.write('module load python3/3.7.2\n')
    f.write(command)
    f.close()
    shellFiles.append(shellScriptFile)
    fileNum += 1

f = open('runMultipleGenerateReports.sh', 'w')
f.write('#!/bin/bash\n')
f.write('\n')
i = 0
for shellFile in shellFiles:
    f.write('sbatch ')
    if i>numJobsPerHour:
        jobDelay = int(i/numJobsPerHour)
        f.write('--begin=now+'+str(jobDelay)+'hour ')
    f.write(shellFile+'\n')
    i+=1
f.close()
