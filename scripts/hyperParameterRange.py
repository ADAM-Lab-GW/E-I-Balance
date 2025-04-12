from surrogateParams import surrParams
import surrogateTraining
import math
import simLogger

outputFile = "learningRate10.csv"
params= surrParams()
f= open(outputFile,"w")
f.write("LEARNING RATE TESTING"+'\n')
f.close()

increments = [1.0,2.5,5.0,7.5]
for i in range(20):
    simTag = 'LR_SGD_'+str(i+40)
    params.simTag = simTag
    params.lr = increments[i%4] * (10**(-1*(math.floor(i/4)+2)))
    acc_hist, loss_hist = surrogateTraining.main(params)

    f= open(outputFile,"a")
    f.write("learningRate,"+str(params.lr)+'\n')
    for e in range(len(acc_hist)):
        f.write(str(acc_hist[e])+','+str(loss_hist[e])+'\n')
    f.close()
      
    notes = 'learning rate testing on bounded spytorch network with LR of ' + str(params.lr)
    inputs = {'surrParams':params,
                'learnRate':str(params.lr),
                'paramVer':str(params.version),
                'optimizer':'SGD'}
    outputs = {'maxAcc':str(max(acc_hist)), 
                'minLoss':str(min(loss_hist)), 
                'accList':acc_hist,
                'lossList':loss_hist}
    simLogger.saveSimulation(simTag, inputs, outputs, notes)
