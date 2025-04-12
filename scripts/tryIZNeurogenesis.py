import surrogateTraining
from surrogateParams import surrParams
import simLogger

parameter_name = "CA3 Pyramidal Subtype 3"
k=0.7996081098116854
a=0.005872238026714838
b=-42.5524776883928
d=588.0
C=41.0
vRest=-55.361334402524086
vThresh=-20.08942170607665
vMin=-38.8680990294091
vPeak=35.8614648558726

parameter_name = "DG middle-age granule"
a=0.001
b=0.2
k=0.06
d=16
C=43.75
vRest=-78.1
vThresh=-40.6
vPeak=38.2
vMin=-61.24

parameter_name = "DG young-age granule"
a=0.0006
b=0.56
k=0.08
d=74.0
C=30.6
vRest=-75.6
vThresh=-42.8
vPeak=38.2
vMin=-66.47

params = surrParams()
params.simTag = 'testIZ'
nIdMin = 0
nIdMax = 50
isExc = True
params.setIZParams(C,k,vMin,vRest,vThresh,vPeak,a,b,d,isExc,nIdMin,nIdMax)


parameter_name = "DG young-age granule"
a=0.0006
b=0.56
k=0.08
d=74.0
C=30.6
vRest=-75.6
vThresh=-42.8
vPeak=38.2
vMin=-66.47
nIdMin = 50
nIdMax = 100
isExc = True
params.setIZParams(C,k,vMin,vRest,vThresh,vPeak,a,b,d,isExc,nIdMin,nIdMax)
params.debug = False

params.sparse_w1=True
params.sparsity_w1=0.9
params.std_w1=0.015
params.sparse_w2=True
params.sparsity_w2=0.5
params.std_w2=0.1

params.lr=0.00001

params.nb_outputs=10

simLogger.logNotes('SETUP LOGGER')

for i in range(1):
    params.simTag = 'DG-young-03_t'+str(i)
    acc,loss = surrogateTraining.main(params)
    
    inputs = {'surrParams':params}
    outputs = {'maxAcc':str(max(acc)),
                'minLoss':str(min(loss)),
                'accList':acc,
                'lossList':loss}
    notes = 'DG young only aged granule cells'
    simLogger.saveSimulation(params.simTag, inputs, outputs, notes)
    
    print(params.simTag + ' TRAINING COMPLETE')
    print(acc)
    print(loss)
