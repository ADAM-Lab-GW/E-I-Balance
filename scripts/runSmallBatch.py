from surrogateParams import surrParams
import surrogateTraining
import math
import simLogger

params= surrParams()

increments = [1.0,2.5,5.0,7.5]
for i in range(20):
    for t in range(10):
        simTag = 'PEG_lr'+str(i+20)+'_t'+str(t)
        params.simTag = simTag
        params.lr = increments[i%4] * (10**(-1*(math.floor(i/4)+2)))
        acc_list, loss_list = surrogateTraining.main(params)
          
        inputs = {'surrparams':params,
        	'learningRate':str(params.lr),
        	'batchSize':str(params.batch_size),
        	'paramVer':str(params.version),
        	'simulator':'spytorch',
        	'dataset':'FashionMNIST'}
        outputs = {'maxAcc':str(max(acc_list)),
        	'minLoss':str(min(loss_list)),
        	'accList':acc_list,
        	'lossList':loss_list}
        notes = 'sample run on pegasus of bounded LIF 2 class FashionMNIST with learning rate of 0.0001'
        simLogger.saveSimulation(simTag, inputs, outputs, notes)
