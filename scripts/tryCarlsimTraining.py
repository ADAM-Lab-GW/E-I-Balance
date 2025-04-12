import surrogateTraining
from surrogateParams import surrParams
import simLogger

params = surrParams()
params.simTag = 'testCarlsim2'
params.setLIFParams(-80.0,-65.0,0.005)
(w1,w2) = simLogger.loadCarlsimWeights('testing61_init_weights.pkl','./data/carlsim')
w1 = w1*50.0
w2 = w2*50.0
(w1,w2) = surrogateTraining.boundWeights(w1,w2,params)

params.debug = True
acc,loss = surrogateTraining.main(params,w1=w1,w2=w2)
print(acc)
print(loss)
