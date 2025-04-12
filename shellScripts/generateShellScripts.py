

baseFileName = './shellScripts/LIF_TEST'
baseSimTag = 'LIF_2_CLASS_'


def generateCommand(trial_num=0,
        num_trials=10,
        learning_rate=0.001,
        sim_tag='default',
        exc_hid_rat=0.5,
        epochs=30,
        log_file='example.log',
        obj_dir='./data/obj',
        w1_std=-1,
        w1_sparse=-1,
        w2_std=-1,
        w2_sparse=-1,
        lif_vRest=0.0,
        lif_vPeak=1.0,
        lif_tau=0.005,
        num_classes=2):
    command = 'python3.7 scripts/excInhTraining.py'
    command += ' -t ' + str(trial_num)
    command += ' -n ' + str(num_trials)
    command += ' -l ' + str(learning_rate)
    command += ' -s ' + sim_tag
    command += ' -r ' + str(exc_hid_rat)
    command += ' -e ' + str(epochs)
    command += ' -f ' + log_file
    command += ' -o ' + obj_dir
    command += ' -w1std ' + str(w1_std)
    command += ' -w1sparse ' + str(w1_sparse)
    command += ' -w2std ' + str(w2_std)
    command += ' -w2sparse ' + str(w2_sparse)
    command += ' --lif_vRest ' + str(lif_vRest)
    command += ' --lif_vPeak ' + str(lif_vPeak)
    command += ' --lif_tau ' + str(lif_tau)
    command += ' -c ' + str(num_classes)
    return command

fileNum = 0
shellFiles = []
simTagList = []
for weight_sparsity in [0.5, 0.75, 0.9]:
    for weight_std in [0.4, 0.55, 0.7]:
        for trial in [0, 2]:
            simTag = baseSimTag + str(fileNum)
            simTagList.append(simTag)
            command = generateCommand(trial_num=trial,
                    num_trials=2,
                    sim_tag=simTag,
                    log_file=simTag+'.log',
                    obj_dir='./data/'+simTag,
                    w1_std=weight_std,
                    w1_sparse=weight_sparsity,
                    learning_rate=0.001,
                    exc_hid_rat = 0.5,
                    lif_vRest = -80.0,
                    lif_vPeak = -65.0,
                    lif_tau = 0.005,
                    num_classes = 2)
            
            f = open(baseFileName+str(fileNum)+'.sh', 'w')
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
            shellFiles.append(baseFileName+str(fileNum)+'.sh')
            fileNum += 1

f = open('runLIF_2_class.sh', 'w')
f.write('#!/bin/bash\n')
f.write('\n')
for shellFile in shellFiles:
    f.write('sbatch '+shellFile+'\n')
f.close()

objFolder = './data/obj'
finalFolder = './data/LIF_2_class'
f = open('moveLIFObjFiles.sh','w')
f.write('#!/bin/bash\n')
f.write('\n')
for simTag in simTagList:
    f.write('mv '+objFolder+'/'+simTag+'*.* '+finalFolder+'\n')
f.close()
