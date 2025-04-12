

baseFileName = './shellScripts/sparse100_10class_noise_'
baseSimTag = 'sparse100_10class_noise_'


def generateCommand(trial_num=0,
        num_trials=10,
        learning_rate=0.001,
        sim_tag='default',
        exc_hid_rat=0.5,
        epochs=30,
        log_file='example.log',
        w1_std=-1,
        w1_sparse=-1,
        w2_std=-1,
        w2_sparse=-1,
        noise_std=[-1,-1]):
    command = 'python3.7 scripts/excInhTraining.py'
    command += ' -t ' + str(trial_num)
    command += ' -n ' + str(num_trials)
    command += ' -l ' + str(learning_rate)
    command += ' -s ' + sim_tag
    command += ' -r ' + str(exc_hid_rat)
    command += ' -e ' + str(epochs)
    command += ' -f ' + log_file
    command += ' -w1std ' + str(w1_std)
    command += ' -w1sparse ' + str(w1_sparse)
    command += ' -w2std ' + str(w2_std)
    command += ' -w2sparse ' + str(w2_sparse)
    command += ' --noise_std ' + str(noise_std[0]) + ' ' + str(noise_std[1])
    return command

fileNum = 200
shellFiles = []
for excRat in [1.00]:
    for lr in [0.01]:
        for w1_std in [0.001]:
            #for noise in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]:
            for noise in [0.00006, 0.00007, 0.00008, 0.00009, 0.0002, 0.0003, 0.0004]:
                for trial in [0,1,2,3,4,5,6,7]:
                    simTag = baseSimTag + str(fileNum)
                    command = generateCommand(trial_num=trial,
                            num_trials=1,
                            sim_tag=simTag,
                            log_file=simTag+'.log',
                            w1_std=w1_std,
                            w1_sparse=0.0,
                            exc_hid_rat=excRat,
                            learning_rate=lr,
                            noise_std=[noise,noise])
                    
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

f = open(baseFileName+'runAll.sh', 'w')
f.write('#!/bin/bash\n')
f.write('\n')
for shellFile in shellFiles:
    f.write('sbatch '+shellFile+'\n')
f.close()
