

baseFileName = './shellScripts/SHD/SHD_TEST_'
baseSimTag = 'SHD_'


def generateCommand(
        sim_tag='default',
        log_file='example.log',
        obj_dir='./data/obj',
        w1_std=-1,
        w1_sparse=-1,
        exc_hid_rat=0.5):
    command = 'python3.7 scripts/paramSHDTraining.py'
    command += ' --sim_tag ' + sim_tag
    command += ' --log_file ' + log_file
    command += ' --obj_dir ' + obj_dir
    command += ' --std_w1 ' + str(w1_std)
    command += ' --sparse_w1 ' + str(w1_sparse)
    command += ' --exc_hid_rat ' + str(exc_hid_rat)
    return command

fileNum = 200
shellFiles = []
simTagList = []
for exc_hid_rat in [0.5, 0.8, 0.95]:
    for weight_sparsity in [0.0]:
        for weight_std in [0.000075, 0.0002, 0.0003, 0.0004, 0.00075]:
        #for weight_std in [0.00005, 0.0001, 0.0005, 0.001]:
            for trial in [4,5,6,7]:
                simTag = baseSimTag + str(int(exc_hid_rat*100)) +'_t'+ str(fileNum)
                simTagList.append(simTag)
                command = generateCommand(
                        sim_tag=simTag,
                        log_file=simTag+'.log',
                        obj_dir='./data/'+simTag,
                        w1_std=weight_std,
                        w1_sparse=weight_sparsity,
                        exc_hid_rat=exc_hid_rat)
                
                f = open(baseFileName+str(fileNum)+'.sh', 'w')
                f.write('#!/bin/bash \n')
                f.write('\n')
                f.write('#SBATCH -o LOG%j.out\n')
                f.write('#SBATCH -e LOG%j.out\n')
                f.write('#SBATCH -p tiny\n')
                f.write('#SBATCH -N 1\n')
                f.write('#SBATCH -D /lustre/groups/adamgrp/repos/surrogate-learning\n')
                f.write('#SBATCH -J Surr_training\n')
                f.write('#SBATCH --export=NONE\n')
                f.write('#SBATCH -t 1:59:59\n')
                f.write('#SBATCH --nice=100\n')
                f.write('\n')
                f.write('module load python3/3.7.2\n')
                f.write(command)
                f.close()
                shellFiles.append(baseFileName+str(fileNum)+'.sh')
                fileNum += 1

f = open('runAllSHD.sh', 'w')
f.write('#!/bin/bash\n')
f.write('\n')
for shellFile in shellFiles:
    f.write('sbatch '+shellFile+'\n')
f.close()
