import surrogateTraining
from surrogateParams import surrParams
import simLogger

params = surrParams()
params.excHidRat = 0.95
params.nb_epochs = 20
params.sparse_w1 = True
params.sparsity_w1 = 0.0
params.std_w1 = 0.001
params.isNoisy = True
params.w1NoiseSTD = 0.00001
params.w2NoiseSTD = 0.00001
params.simTag = 'testNoise7'
simLogger.setupLogger(fileName=params.simTag+'.log', customObjDir='./data/'+params.simTag+'/obj')

acc,loss = surrogateTraining.main(params)
print(acc)
print(loss)

simLogger.saveSimulation(params.simTag, {'surrParams':params}, {'accList':acc,'lossList':loss,'maxAcc':str(max(acc)),'minLoss':str(min(loss))}, notes='Testing with noisy weight updates')
