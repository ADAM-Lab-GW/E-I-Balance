import surrogateTraining
from surrogateParams import surrParams
import simLogger
import argparse
from weightAnalysis import getWeightFromUniqueId

parser = argparse.ArgumentParser(description='Training program for a range of parameters with the SHD dataset')
parser.add_argument('--log_file', help='log file path')
parser.add_argument('--obj_dir', help='director for object files')
parser.add_argument('--sim_tag', help='simTag for the simulation')
parser.add_argument('--std_w1', help='input to hidden standard deviation', type=float)
parser.add_argument('--sparse_w1',help='input to hidden sparsity', type=float)
parser.add_argument('--exc_hid_rat',help='percent hidden layer excitatory', type=float)
parser.add_argument('--epochs', help='number of epochs to run', type=int, default=200)
parser.add_argument('--start_epoch',help='starting epoch, do not include this parameter if starting from init. Example: --start_epoch 5, will load _e4 weights and start with epoch 5', type=int, default=-2)
parser.add_argument('--learn_rate',help='learning rate', type=float, default=0.001)
args = parser.parse_args()

params = surrParams()
params.simTag =args.sim_tag
params.setLIFParams(0.0,1.0,0.005)
params.sparse_w1=True
params.sparsity_w1=args.sparse_w1
params.std_w1=args.std_w1
params.sparse_w2=True
params.sparsity_w2=0.0
params.std_w2=0.0001
params.excHidRat=args.exc_hid_rat
params.debug = False
params.nb_inputs=700
params.nb_hidden=200
params.nb_outputs=20
params.nb_epochs=args.epochs

params.batch_size=256
params.data_set="SHD"
params.lr=args.learn_rate

simLogger.setupLogger(fileName=args.log_file, customObjDir=args.obj_dir)
acc=[]
loss=[]
if(args.start_epoch != -2):
    params.start_epoch=args.start_epoch
    w1Tag = params.simTag+'_w1_e'+str(params.start_epoch-1)
    w2Tag = params.simTag+'_w2_e'+str(params.start_epoch-1)
    if(params.start_epoch==-1):
        w1Tag = params.simTag+'_w1_init'
        w2Tag = params.simTag+'_w2_init'
    w1 = getWeightFromUniqueId(w1Tag, args.obj_dir)
    w2 = getWeightFromUniqueId(w2Tag, args.obj_dir)
    acc,loss = surrogateTraining.main(params, w1, w2)
else:
    acc,loss = surrogateTraining.main(params)
print(acc)
print(loss)
inputs = {'surrParams':params}
outputs = {'maxAcc':str(max(acc)),
            'minLoss':str(min(loss)),
            'accList':acc,
            'lossList':loss}
notes = 'Test for Spiking Heidelberg Digits Dataset'
simLogger.saveSimulation(params.simTag, inputs, outputs, notes)
