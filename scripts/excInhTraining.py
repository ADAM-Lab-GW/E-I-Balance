# This script takes in a command line args for the exc/inh ratio
#  in the hidden layer
from surrogateParams import surrParams
import simLogger
import argparse
import surrogateTraining
from weightAnalysis import getWeightFromUniqueId

# collect command line args (all have defaults)
parser = argparse.ArgumentParser(description='Training program for a particular learning rate')
parser.add_argument('-n', '--num_trials', help='Number of trials to run, where a trial is an exact repeat of the hyperparameters', default=3, type=int)
parser.add_argument('-t', '--trial_num', help='Starting trial number for this set of trials', default=1, type=int)
parser.add_argument('-l', '--learning_rate', help='Learning rate for the training', default=0.0001, type=float)
parser.add_argument('-s', '--sim_tag', help='Simtag for the simulation', default='defaultSimtag')
parser.add_argument('-r', '--exc_hid_rat', help='Ratio of excitatory neurons in the hidden layer', default=0.5, type=float)
parser.add_argument('-e', '--epochs', help='Number of epochs for each trial', default=30, type=int)
parser.add_argument('-f', '--log_file', help='Filepath for the log file this will use', default='example.log')
parser.add_argument('-o', '--obj_dir', help='Object directory', default='./data/obj')
parser.add_argument('-w1std', '--w1_std', help='Input-Hidden initial weight distribution', default=-1,type=float)
parser.add_argument('-w1sparse', '--w1_sparse', help='Input-Hidden initial sparsity', default=-1, type=float)
parser.add_argument('-w2std', '--w2_std', help='Hidden-Output initial weight distribution', default=-1,type=float)
parser.add_argument('-w2sparse', '--w2_sparse', help='Hidden-Output initial sparsity', default=-1, type=float)
parser.add_argument('--lif_vRest', help='Custom LIF vRest', default=0.0, type=float)
parser.add_argument('--lif_vPeak', help='Custom LIF vPeak', default=1.0, type=float)
parser.add_argument('--lif_tau', help='Custom LIF membrane time constant', default=0.005, type=float)
parser.add_argument('-c', '--class_num', help='Number of classes being tested', default=10, type=int)
parser.add_argument('--w1_pre_load_id',help='Load input to hidden weights from file with unique id', default=None)
parser.add_argument('--w1_pre_load_folder',help='Load input to hidden weights from file. This is the object folder', default=None)
parser.add_argument('--w2_pre_load_id',help='Load hidden to output weights from file with unique id', default=None)
parser.add_argument('--w2_pre_load_folder',help='Load hidden to output weights from file. This is the object folder', default=None)
parser.add_argument('--noise_std', help='In-Hid and Hid-Out weight update noise STD', default=[-1,-1], nargs=2, type=float)
parser.add_argument('--debug', help='Turn on debug mode', action='store_true')

args = parser.parse_args()

params = surrParams()
params.lr = args.learning_rate
params.excHidRat = args.exc_hid_rat
params.nb_epochs = args.epochs
params.nb_outputs = args. class_num

if(args.w1_sparse != -1 and args.w1_std != -1):
    params.sparse_w1 = True
    params.std_w1 = args.w1_std
    params.sparsity_w1 = args.w1_sparse

if(args.w2_sparse != -1 and args.w2_std != -1):
    params.sparse_w2 = True
    params.std_w2 = args.w2_std
    params.sparsity_w2 = args.w2_sparse

params.setLIFParams(args.lif_vRest, args.lif_vPeak, args.lif_tau)

if(args.noise_std[0] != -1):
    params.isNoisy = True
    params.w1NoiseSTD = args.noise_std[0]
    params.w2NoiseSTD = args.noise_std[1]

w1 = None
w2 = None
if(args.w1_pre_load_id!=None):
    objFolder='./data/obj' if args.w1_pre_load_folder==None else args.w1_pre_load_folder
    w1 = getWeightFromUniqueId(args.w1_pre_load_id, objFolder=objFolder)
if(args.w2_pre_load_id!=None):
    objFolder='./data/obj' if args.w2_pre_load_folder==None else args.w2_pre_load_folder
    w2 = getWeightFromUniqueId(args.w2_pre_load_id, objFolder=objFolder)

if(args.debug):
    params.debug = True

simLogger.setupLogger(fileName=args.log_file, customObjDir=args.obj_dir)

for t in range(args.trial_num, args.trial_num+args.num_trials):
    params.simTag = args.sim_tag + '_t'+str(t)
    notes = 'using standard gradient descent learning rate testing on bounded spytorch network with LR of ' + str(params.lr) + ' on a network with a hidden layer that is ' + str(params.excHidRat) + ' excitatory'
    inputs = {'surrParams':params,
                'learnRate':str(params.lr),
                'excHidRat':str(params.excHidRat),
                'paramVer':str(params.version),
                'sparse_w1':str(params.sparse_w1),
                'sparsity_w1':str(params.sparsity_w1),
                'std_w1':str(params.std_w1),
                'sparse_w2':str(params.sparse_w2),
                'sparsity_w2':str(params.sparsity_w2),
                'std_w2':str(params.std_w2),
                'lif_vRest':str(params.vRest),
                'lif_vPeak':str(params.vPeak),
                'lif_tau_mem':str(params.tau_mem),
                'w1_pre_load_id':str(args.w1_pre_load_id),
                'w2_pre_load_id':str(args.w2_pre_load_id),
                'dataset':'fashionMNIST'}

    if w1==None and w2==None:
        acc_list, loss_list = surrogateTraining.main(params)
    else:
        acc_list, loss_list = surrogateTraining.main(params, w1=w1, w2=w2)

    outputs = {'maxAcc':str(max(acc_list)), 
                'minLoss':str(min(loss_list)), 
                'accList':acc_list,
                'lossList':loss_list}
    simLogger.saveSimulation(params.simTag, inputs, outputs, notes)
