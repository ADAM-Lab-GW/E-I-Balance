import simLogger
from surrogateParams import surrParams
import surrogateTraining

simTag = 'test_simulation5'
notes = 'Another test for using auto save on non string objects'

params = surrParams()
params.simTag = simTag
acc_list, loss_list = surrogateTraining.main(params)
#acc_list, loss_list = [0.6,0.85,0.8],[0.8,0.1,0.2]
maxAcc = max(acc_list)
minLoss = min(loss_list)

# SAVING SIMUALTION
# input and output dictionary list the names of the variable and the objects for that var
# by making these mappings, if wanted, the vars can be made to strings and won't be saved
#  to file and will just be output to the log file
# additional notes can be listed for any other thoughts that are useful to keep with the simulation
# often times certain vars are useful to pull out and list directly in the log file so they can
#  be read immediately, and the min and max loss are examples of that here
inputs = {'surrParams':params}
outputs = {'maxAcc':str(maxAcc),'minLoss':str(minLoss),'acc_list':acc_list,'loss_list':loss_list}
simLogger.saveSimulation(simTag, inputs, outputs, notes)
