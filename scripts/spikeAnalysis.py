import argparse
from configparser import Interpolation
import weightAnalysis
from surrogateParams import surrParams
import surrogateTraining
import torch
import numpy as np
import matplotlib.pyplot as plt

classes = {0:'T-shirt/top',
            1:'Trouser',
            2:'Pullover',
            3:'Dress',
            4:'Coat',
            5:'Sandal',
            6:'Shirt',
            7:'Sneaker',
            8:'Bag',
            9:'Ankle Boot'}

def calcInputHiddenCov(inputSpk, hiddenSpk):
    # input and hidden matrices have dimension 1 as neuron
    #  dimension 2 is the the time step
    # Each vector for the neurons should be 0 or 1

    # for covariance calculations, we need zero mean vectors
    for i in range(784):
        avg = np.average(inputSpk[i])
        if(avg != 0):
            inputSpk[i] = inputSpk[i]/(avg)

    for i in range(100):
        avg = np.average(hiddenSpk[i])
        if(avg != 0):
            hiddenSpk[i] = hiddenSpk[i]/(avg)

    # we can then align and combine the spike vectors
    #  by moving the hidden layer spikes one time step forward
    #  they will align with the input layer
    fullSpk = np.concatenate((inputSpk[:,:99],hiddenSpk[:,1:]))

    # calculate the covariance of the entire spike matrix
    #  this will also create covariance between all neurons in the
    #  input and hidden layer, and we can choose to pull out the 
    #  specific input to hidden layer covariance
    covMat = np.cov(fullSpk)

    # we are returning the specific input-hidden covariance matrix
    #  and the full covariance matrix, so users can don't need to 
    #  pull apart the matrix themselves if they don't want to
    return covMat[:784,784:], covMat

def calcInputOutputCov(inputSpk, outputVolt):
    # input and output matrices have dimension 1 as neuron
    #  dimension 2 is the the time step
    # Each vector for the neurons should be 0 or 1 for firing
    #  the output vectors will have voltage values

    # for covariance calculations, we need zero mean vectors
    for i in range(784):
        avg = np.average(inputSpk[i])
        if(avg != 0):
            inputSpk[i] = inputSpk[i]/(avg)

    for i in range(2):
        avg = np.average(outputVolt[i])
        if(avg != 0):
            outputVolt[i] = outputVolt[i]/(avg)

    # we can then align and combine the spike vectors
    #  by moving the output layer spikes two time step forward
    #  they will align with the input layer
    fullSpk = np.concatenate((inputSpk[:,0:98],outputVolt[:,2:100]))

    # calculate the covariance of the entire spike matrix
    #  this will also create covariance between all neurons in the
    #  input and output layer, and we can choose to pull out the 
    #  specific input to output layer covariance
    covMat = np.cov(fullSpk)

    # we are returning the specific input-output covariance matrix
    #  and the full covariance matrix, so users can don't need to 
    #  pull apart the matrix themselves if they don't want to
    return covMat[:784,784:], covMat

def plotInputHidCov(inputHidCov, title='Sample Covariance', filePath='./heatmaps/sampleCov.png'):
    vmin = np.min(inputHidCov[:,:])
    vmax = np.max(inputHidCov[:,:])
    fig, axs = plt.subplots(10,10)
    for i in range(10):
        for j in range(10):
            hidMap = inputHidCov[:,i*10 + j].reshape(28,28)
            im = axs[i][j].imshow(hidMap, interpolation='none', vmin=vmin, vmax=vmax)
            axs[i][j].get_xaxis().set_visible(False)
            axs[i][j].get_yaxis().set_visible(False)

    fig.colorbar(im, ax=axs[:,:], orientation='vertical')
    fig.suptitle(title, y=0.92, fontsize=18)
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)

    fig.set_size_inches(10, 10)
    plt.savefig(filePath, dpi=200, transparent=False)

def plotInputOutputCov(inputOutputCov, title='Sample Covariance', filePath='./heatmaps/sampleInOutCov.png'):
    fig, axs = plt.subplots(1, 3)
    vmin = np.min(inputOutputCov[:,0:1])
    vmax = np.max(inputOutputCov[:,0:1])
    for i in range(2):
        inMap = inputHidCov[:,i].reshape(28,28)
        im = axs[i].imshow(inMap, interpolation='none', vmin=vmin, vmax=vmax)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        axs[i].title.set_text('Class ' + str(i) + ': ' + classes[i])

    fig.colorbar(im, ax=axs[0:2], orientation='horizontal')

    diffMap = inputOutputCov[:,0].reshape(28,28)-inputOutputCov[:,1].reshape(28,28)
    vmin = np.min(diffMap)
    vmax = np.max(diffMap)
    im = axs[2].imshow(diffMap, interpolation='none', vmin=vmin, vmax=vmax)
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)
    axs[2].title.set_text('Class 0 - Class 1')

    fig.colorbar(im, ax=axs[2], orientation='horizontal')
    fig.suptitle(title, y=0.8, fontsize=18)
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)

    fig.set_size_inches(10, 7)
    plt.savefig(filePath, dpi=200, transparent=False)

def batchSpikeCounts(spk, neuronRange=None):
    # spk is a batch of either input or hidden spiking data
    #  we are going to aggregate the data at each level

    timeStepTotal = []  # total spikes at each time step
                        #  first dimension - trial number
                        #  second dimension - time step

    for i in range(len(spk)):
        trialSpk = spk[i]
        temp = []
        for t in range(len(trialSpk)):
            # build the temp array one time step at a time
            if(neuronRange==None):
                temp.append(np.sum(trialSpk[t,:]))
            else:
                temp.append(np.sum(trialSpk[t,neuronRange]))


        # the temp array will contain all the time step totals
        #  for that particular i'th trial (repeat for each trial)
        timeStepTotal.append(temp)
    
    return timeStepTotal

def plotSpikeCount(timeStepTotal, neuronRange=None, title='Default', filePath='./spikeActivity.png'):
    if(neuronRange == None):
        aggregate = np.sum(timeStepTotal, axis=0)
    else:
        aggregate = np.sum(timeStepTotal[neuronRange,:], axis=0)
    
    fig, axs = plt.subplots(1,1)
    fig.suptitle("Aggregate Activity - "+title)
    axs.plot(aggregate)
    axs.set_box_aspect(1)

    axs.set_ylabel("Total Batch Spikes")
    axs.set_xlabel("Time step (ms)")

    plt.savefig(filePath, dpi=400)

if(__name__=='__main__'):
    parser = argparse.ArgumentParser(description='Training program for a particular learning rate')
    parser.add_argument('-w1', '--w1_tag', help='Simtag for the input to hidden matrix pickle file', default='defaultSimtag')
    parser.add_argument('-w2', '--w2_tag', help='Simtag for the hidden to output matrix pickle file', default='defaultSimtag')
    parser.add_argument('-t', '--tag', help='Tag for images and file names', default='default')
    parser.add_argument('-p', '--plot_dir', help='Directory for where the plots should be saved', default='.')
    parser.add_argument('-o', '--obj_path', help='Folder path for the object directory containing the weight files', default='./data/obj')
    parser.add_argument('-s', '--sample_id', help='Sample ID number from the testing set', default=0, type=int)
    args = parser.parse_args()

    w1 = weightAnalysis.getWeightFromUniqueId(args.w1_tag, objFolder=args.obj_path)
    w2 = weightAnalysis.getWeightFromUniqueId(args.w2_tag, objFolder=args.obj_path)

    params = surrParams()
    x_train, x_test, y_train, y_test = surrogateTraining.getDataSet()
    x_batch, y_batch, batch_index = surrogateTraining.get_mini_batch(x_test, y_test, params, shuffle=False)
    output, other_recordings = surrogateTraining.run_snn(x_batch.to_dense(), w1, w2, params)
    mem_rec, spk_rec = other_recordings

    sampleID = args.sample_id
    inputSpk = x_batch[sampleID].to_dense().numpy().T
    hiddenSpk = spk_rec[sampleID].detach().numpy().T
    outputSpk = output[sampleID].detach().numpy().T

    inputHidCov, fullCov = calcInputHiddenCov(inputSpk, hiddenSpk)
    plotInputHidCov(inputHidCov, title=args.tag + '_s'+str(args.sample_id)+' Input-Hidden Covariance', filePath='./heatmaps/inputHidden_'+args.tag+'_s'+str(args.sample_id)+'.png')

    inputOutCov, fullCov = calcInputOutputCov(inputSpk, outputSpk)
    plotInputOutputCov(inputOutCov, title=args.tag + '_s'+str(args.sample_id)+' Input-Output Covariance', filePath='./heatmaps/inputOutput_'+args.tag+'_s'+str(args.sample_id)+'.png')

    inputHidCovAvg = np.zeros(inputHidCov.shape)
    inputOutCovAvg = np.zeros(inputOutCov.shape)
    for t in range(len(x_batch)):
        inputSpk = x_batch[t].to_dense().numpy().T
        hiddenSpk = spk_rec[t].detach().numpy().T
        outputSpk = output[t].detach().numpy().T
        inputHidCov, fullCov = calcInputHiddenCov(inputSpk, hiddenSpk)
        inputHidCovAvg += inputHidCov
        inputOutCov, fullCov = calcInputOutputCov(inputSpk, outputSpk)
        inputOutCovAvg += inputOutCov
    inputHidCovAvg /= len(x_batch)
    inputOutCovAvg /= len(x_batch)
    plotInputHidCov(inputHidCovAvg, title=args.tag + ' Input-Hidden Covariance Batch AVG', filePath='./heatmaps/inputHiddenAVG_'+args.tag+'_batch.png')
    plotInputOutputCov(inputOutCovAvg, title=args.tag + ' Input-Output Covariance Batch AVG', filePath='./heatmaps/inputOutputAVG_'+args.tag+'_batch.png')

    inputSpkCounts = batchSpikeCounts(x_batch.to_dense().numpy())
    hiddenSpkCounts = batchSpikeCounts(spk_rec.detach().numpy())

    plotSpikeCount(inputSpkCounts, title='Input '+args.tag, filePath='./activityPlots/activity_input_'+args.tag+'.png')
    plotSpikeCount(hiddenSpkCounts, title='Hidden '+args.tag, filePath='./activityPlots/activity_hidden_'+args.tag+'.png')