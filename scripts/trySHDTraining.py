import surrogateTraining
from surrogateParams import surrParams
import simLogger

params = surrParams()
params.simTag = 'testSHD01'
params.setLIFParams(0.0,1.0,0.005)
params.sparse_w1=True
params.sparsity_w1=0.75
params.std_w1=0.0004
params.sparse_w2=True
params.sparsity_w2=0.0
params.std_w2=0.0001
params.debug = False
params.nb_inputs=700
params.nb_hidden=200
params.nb_outputs=20
params.nb_epochs=200
params.batch_size=256
params.data_set="SHD"
params.lr=1e-3

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
