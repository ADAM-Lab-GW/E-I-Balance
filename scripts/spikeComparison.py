import spikeAnalysis
import weightAnalysis
import surrogateTraining
from surrogateParams import surrParams

def compareTrainingCovariance(simTag, objFolder='./data/obj'):
    w1A = weightAnalysis.getWeightFromUniqueId(simTag+'_w1_init', objFolder=objFolder)
    w2A = weightAnalysis.getWeightFromUniqueId(simTag+'_w2_init', objFolder=objFolder)
    w1B = weightAnalysis.getWeightFromUniqueId(simTag+'_w1_e29_b45', objFolder=objFolder)
    w2B = weightAnalysis.getWeightFromUniqueId(simTag+'_w2_e29_b45', objFolder=objFolder)

    params = surrParams()
    x_train, x_test, y_train, y_test = surrogateTraining.getDataSet()
    x_batch, y_batch, batch_index = surrogateTraining.get_mini_batch(x_test, y_test, params, shuffle=False)
    output, other_recordings = surrogateTraining.run_snn(x_batch.to_dense(), w1A, w2A, params)
    mem_rec, spk_rec = other_recordings

    for sampleID in range(params.batch_size):
        inputSpk = x_batch[sampleID].to_dense().numpy().T
        hiddenSpk = spk_rec[sampleID].detach().numpy().T
        inputHidCov, fullCov = spikeAnalysis.calcInputHiddenCov(inputSpk, hiddenSpk)

        