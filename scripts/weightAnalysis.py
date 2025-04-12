import pickle
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import random

def getWeightFromUniqueId(uniqueId, objFolder='./data/obj'):
    print('ATTEMPTING TO LOAD: '+objFolder+'/'+uniqueId)
    fileList = os.listdir(objFolder)
    filePath = [i for i in fileList if uniqueId in i][0]
    with open(objFolder+'/'+filePath, "rb") as f:
        tempWeight = pickle.load(f)
    
    return tempWeight


def w1Hist(w1, uniqueId, simTag):
    fig, axs = plt.subplots(1,1)
    axs.set_title('Input to Hidden Weight Distribution '+uniqueId)
    axs.set_xlabel('Weight')
    axs.set_ylabel('Frequency')
    axs.set_xlim([0,0.015])
    axs.set_yscale('log')
    axs.set_ylim([0,10e4])
    axs.hist(w1.flatten().detach().numpy(), bins=21)
    plt.savefig('./plots/' + uniqueId + '.png', dpi=300)
    plt.close()

def w1HistCompare(simTag, dataSet, descriptor):
    baseFolder = '/mnt/l/Grad School/Research/Repos/ICONS-data/'
    dataSetFolder = baseFolder+dataSet
    logFolder = baseFolder+dataSet+'/logs'
    objFolder = baseFolder+dataSet+'/obj'

    print('loading data '+simTag+'...')
    w1Init = getWeightFromUniqueId(simTag+'_w1_init', objFolder=objFolder)
    w1Fin = getWeightFromUniqueId(simTag+'_w1_e29_b234', objFolder=objFolder)

    xmin = 0
    xmax = 0.09
    bins = [x / 1000.0 for x in range(xmin*1000, int(xmax*1000),2)]

    print('generating graph...')
    fig, axs = plt.subplots(1,2)
    axs[0].grid(which='major')
    axs[0].minorticks_on()
    #axs[0].grid(which='minor')
    axs[0].set_axisbelow(True)
    axs[0].set_title('Input to Hidden Weight Distribution '+simTag)
    axs[0].set_xlabel('Weight')
    axs[0].set_ylabel('Frequency')
    axs[0].set_xlim([xmin,xmax])
    axs[0].set_ylim([0,10e4])
    axs[0].set_yscale('symlog')
    axs[0].hist(w1Init.flatten().detach().numpy(), bins=bins)

    axs[1].grid(which='major')
    axs[1].minorticks_on()
    #axs[1].grid(which='minor')
    axs[1].set_axisbelow(True)
    axs[1].set_xlabel('Weight')
    axs[1].set_ylabel('Frequency')
    axs[1].set_xlim([xmin,xmax])
    axs[1].set_yscale('symlog')
    axs[1].set_ylim([0,10e4])
    axs[1].hist(w1Fin.flatten().detach().numpy(), bins=bins)
    plt.tight_layout()

    print('saving graph...')
    plt.savefig(dataSetFolder+'/'+descriptor+'.png', dpi=300)
    plt.savefig(dataSetFolder+'/'+descriptor+'.svg')
    plt.close()    

def getNeuronInputImage(weights, i):
    # generate a 28x28 matrix based on the weights
    #  connecting the ith neuron in the hidden layer
    #  to the INPUT layer
    # weights is a pytorch array
    # returns a numpy array
    weights = weights.detach().numpy()
    return weights[:,i].reshape(28,28)

def generateAllInputToHiddenMaps(weight, figTitle, filePath=None, vmin=-0.005, vmax=0.005):
    fig, axs = plt.subplots(10,10)
    for i in range(10):
        for j in range(10):
            im = axs[i][j].imshow(getNeuronInputImage(weight,i*10+j), vmin=vmin, vmax=vmax, interpolation='none')
            axs[i][j].get_xaxis().set_visible(False)
            axs[i][j].get_yaxis().set_visible(False)

    fig.colorbar(im, ax=axs[:,:], orientation='vertical')
    fig.suptitle(figTitle, y=0.92, fontsize=18)
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if(filePath != None):
        fig.set_size_inches(10, 10)
        plt.savefig(filePath, dpi=200, transparent=False)
    plt.show()

def getWeightTrajectory(layerNum, pre, post, simTag, objFolder='./data/obj'):
    fileList = os.listdir(objFolder)
    print(fileList)
    weight = []
    zeros = []

    # get init weights
    uniqueId = simTag+'_w'+str(layerNum)+"_init"
    filePath = [i for i in fileList if uniqueId in i][0]
    print(filePath)
    with open(objFolder+'/'+filePath, "rb") as f:
        tempWeight = pickle.load(f)
    weight.append(tempWeight[pre,post].detach().numpy())
    zeros.append(78400 - np.count_nonzero(tempWeight.detach().numpy()))

    # get weights across each epoch
    for epoch in range(30):
        print("loading epoch " + str(epoch))
        #for batchNum in range(46):
        for batchNum in [234]:
            uniqueId = simTag+'_w'+str(layerNum)+"_e"+str(epoch)+"_b"+str(batchNum)
            filePath = [i for i in fileList if uniqueId in i][0]
            with open(objFolder+'/'+filePath, "rb") as f:
                tempWeight = pickle.load(f)
            
            if (batchNum==0):
                print(tempWeight.grad)

            #print(tempWeight[pre,post].detach().numpy())
            weight.append(tempWeight[pre,post].detach().numpy())
            zeros.append(78400 - np.count_nonzero(tempWeight.detach().numpy()))

    return weight, zeros

def getPegasusWeightTrajectoriesOutput():
    layer = 2
    for t in range(10):
        simTag = 'PEG_lr8_t'+str(t)
        pre = []
        for i in range(100):
            pre.append(i)
        w0 = getWeightTrajectory(layer,pre,0, simTag, objFolder='PEG_lr8/PEG_lr8')
        w1 = getWeightTrajectory(layer,pre,1, simTag, objFolder='PEG_lr8/PEG_lr8')
        #print(w)
        # while(w[0] < 0.001):
        #     i += 1
        #     w = getWeightTrajectory(layer,i,0)

        fig, axs = plt.subplots(1,2, sharey='all')
        fig.tight_layout()
        fig.suptitle("Hidden to Output Weight Trajectories From " + simTag)
        axs[0].set_title('Output to T-shirt')
        axs[0].plot(w0)
        axs[1].set_title('Output to Trousers')
        axs[1].plot(w1)
        axs[0].set_box_aspect(1)
        axs[1].set_box_aspect(1)
        axs[0].set_ylim([-0.3,0.3])
        plt.subplots_adjust(left=.15)
        axs[0].set_ylabel("Weight Value")
        axs[0].set_xlabel("Update number (30 epochs total)")
        axs[1].set_xlabel("Update number (30 epochs total)")
        plt.savefig("./plots/Pegasus Weight Trajectory "+simTag+".png", dpi=400)

def getWeightDistributionsTrials():
    all_weights = []
    for t in range(10):
        simTag = 'PEG_lr8_t' + str(t)
        uniqueId = simTag + '_w1_init'
        w2 = getWeightFromUniqueId(uniqueId, objFolder='./PEG_lr8/PEG_lr8')
        fig, axs = plt.subplots(1,1)
        axs.set_title('Initial Input to Hidden Weight Distribution '+simTag)
        axs.set_xlabel('Weight')
        axs.set_ylabel('Frequency')
        axs.set_xlim([0,0.15])
        axs.set_yscale('log')
        axs.set_ylim([0,10e5])
        axs.hist(w2.flatten().detach().numpy(), bins=21)
        plt.savefig('./plots/' + uniqueId + '.png', dpi=300)

        all_weights.append(w2.tolist())

    print(len(all_weights))
    print(len(all_weights[0]))
    print(len(all_weights[0][0]))
    #print(all_weights)
    fig, axs = plt.subplots(1,1)
    axs.set_title('Initial Hidden to Output Weight Distribution '+simTag)
    axs.set_xlabel('Trial')
    axs.set_ylabel('Weight')
    #axs.set_ylim([-0.3,0.3])
    axs.violinplot(all_weights)
    plt.savefig('./plots/PEG_lr8_w2_violin.png', dpi=300)
    
def getWeightDistributionsTrialsEpochBatch():
    for t in range(10):
        for e in range(30):
            for b in range(46):
                simTag = 'PEG_lr8_t' + str(t)
                uniqueId = simTag + '_w1_e' + str(e) + '_b' + str(b)
                print(uniqueId)
                w2 = getWeightFromUniqueId(uniqueId, objFolder='./PEG_lr8/PEG_lr8')
                w1Hist(w2, uniqueId, simTag)
                # fig, axs = plt.subplots(1,1)
                # axs.set_title('Initial Hidden to Output Weight Distribution '+uniqueId)
                # axs.set_xlabel('Weight')
                # axs.set_ylabel('Frequency')
                # axs.set_xlim([-0.3,0.3])
                # axs.set_ylim([0,140])
                # axs.hist(w2.flatten().detach().numpy(), bins=np.arange(-0.3,0.3,0.04))
                # plt.savefig('./plots/' + uniqueId + '.png', dpi=300)

def getWeightDistribution(simTag):
    uniqueId = simTag + '_w2'
    w2 = getWeightFromUniqueId(uniqueId, objFolder='./data/obj')
    fig, axs = plt.subplots(1,1)
    axs.set_title('Initial Hidden to Output Weight Distribution '+simTag)
    axs.set_xlabel('Weight')
    axs.set_ylabel('Frequency')
    axs.set_xlim([-0.3,0.3])
    axs.set_ylim([0,140])
    axs.hist(w2.flatten().detach().numpy(), bins=np.arange(-0.3,0.3,0.04))
    plt.savefig('./plots/' + uniqueId + '.png', dpi=300)

    print(len(w2))
    print(len(w2[0]))
    print(w2)

def getWeightHeatmaps(simTag):
    for t in range(10):
        uniqueIdInit = simTag + '_t'+str(t)+'_w1_init'
        uniqueIdFinal = simTag + '_t'+str(t)+'_w1_e29_b45'
        wInit = getWeightFromUniqueId(uniqueIdInit, objFolder='./PEG_lr8/PEG_lr8')
        wFinal = getWeightFromUniqueId(uniqueIdFinal, objFolder='./PEG_lr8/PEG_lr8')
        generateAllInputToHiddenMaps(wFinal-wInit, figTitle=simTag+'_t'+str(t) + ' Change in Input to Hidden Weights Across All Training', 
            filePath='./heatmaps/'+simTag+'_t'+str(t)+'_heatmap.png',
            vmin=-0.0005, vmax=0.0005)

#getWeightDistribution('TEST_EXC_HID_RAT_t3')
#getWeightDistributionsTrials()
#getWeightDistributionsTrialsEpochBatch()

#w1 = getWeightFromUniqueId('PEG_lr8_t0_w1_init', objFolder='./PEG_lr8/PEG_lr8')
#w2 = getWeightFromUniqueId('PEG_lr8_t0_w1_e29_b30', objFolder='./PEG_lr8/PEG_lr8')
#generateAllInputToHiddenMaps(w2-w1, figTitle='PEG_lr8_t1 Change in Input to Hidden Weights Across All Training', filePath='./PEG_lr8_t1_w1_update.png')

#getWeightHeatmaps('PEG_lr8')

def plotWeightTraj(simTag, dataSet, descriptor):
    baseFolder = '/mnt/l/Grad School/Research/Repos/ICONS-data/'
    dataSetFolder = baseFolder+dataSet
    logFolder = baseFolder+dataSet+'/logs'
    objFolder = baseFolder+dataSet+'/obj'

    n = 300
    pre = np.empty(n)
    for i in range(n):
        pre[i] = random.randint(0,783)

    post = np.empty(n)
    for i in range(n):
        post[i] = random.randint(0,99)

    tempW, tempZ = getWeightTrajectory(1,pre, post, simTag, objFolder = objFolder)
    fig, axs = plt.subplots(1,1)
    plt.plot(tempW)
    plt.title(simTag)
    plt.ylim([0,0.03])
    axs.set_box_aspect(1)
    plt.savefig(dataSetFolder+'/'+descriptor+'.png',dpi=600)
    plt.savefig(dataSetFolder+'/'+descriptor+'.svg')
    print("Plot Saved")

if __name__ == "__main__":
    descriptor = 'firingRates_minimal_weights'
    simTag = 'sparse050_10class_108_t0'
    dataSet = 'sparse050_10class'
    plotWeightTraj(simTag, dataSet, descriptor)
    simTag = 'sparse080_10class_112_t4'
    dataSet = 'sparse080_10class'
    plotWeightTraj(simTag, dataSet, descriptor)
    simTag = 'sparse095_10class_115_t7'
    dataSet = 'sparse095_10class'
    plotWeightTraj(simTag, dataSet, descriptor)
    simTag = 'sparse100_10class_111_t3'
    dataSet = 'sparse100_10class'
    plotWeightTraj(simTag, dataSet, descriptor)

    descriptor = 'firingRates_10Hz_weights'
    simTag = 'sparse050_10class_124_t0'
    dataSet = 'sparse050_10class'
    plotWeightTraj(simTag, dataSet, descriptor)
    simTag = 'sparse080_10class_126_t2'
    dataSet = 'sparse080_10class'
    plotWeightTraj(simTag, dataSet, descriptor)
    simTag = 'sparse095_10class_124_t0'
    dataSet = 'sparse095_10class'
    plotWeightTraj(simTag, dataSet, descriptor)
    simTag = 'sparse100_10class_130_t6'
    dataSet = 'sparse100_10class'
    plotWeightTraj(simTag, dataSet, descriptor)

    descriptor = 'firingRates_20Hz_weights'
    simTag = 'sparse050_10class_147_t7'
    dataSet = 'sparse050_10class'
    plotWeightTraj(simTag, dataSet, descriptor)
    simTag = 'sparse080_10class_140_t0'
    dataSet = 'sparse080_10class'
    plotWeightTraj(simTag, dataSet, descriptor)
    simTag = 'sparse095_10class_147_t7'
    dataSet = 'sparse095_10class'
    plotWeightTraj(simTag, dataSet, descriptor)
    simTag = 'sparse100_10class_147_t7'
    dataSet = 'sparse100_10class'
    plotWeightTraj(simTag, dataSet, descriptor)


    # w1HistCompare('sparse050_10class_108_t0','sparse050_10class','_weight_hist_min_comparison')
    # w1HistCompare('sparse095_10class_115_t7','sparse095_10class','_weight_hist_min_comparison')
    # w1HistCompare('sparse100_10class_111_t3','sparse100_10class','_weight_hist_min_comparison')
    # w1HistCompare('sparse080_10class_112_t4','sparse080_10class','_weight_hist_min_comparison')

    # w1HistCompare('sparse050_10class_147_t7','sparse050_10class','_weight_hist_20Hz_comparison')
    # w1HistCompare('sparse080_10class_140_t0','sparse080_10class','_weight_hist_20Hz_comparison')
    # w1HistCompare('sparse095_10class_147_t7','sparse095_10class','_weight_hist_20Hz_comparison')
    # w1HistCompare('sparse100_10class_147_t7','sparse100_10class','_weight_hist_20Hz_comparison')

    # w1HistCompare('sparse050_10class_124_t0','sparse050_10class','_weight_hist_10Hz_comparison')
    # w1HistCompare('sparse080_10class_126_t2','sparse080_10class','_weight_hist_10Hz_comparison')
    # w1HistCompare('sparse095_10class_124_t0','sparse095_10class','_weight_hist_10Hz_comparison')
    # w1HistCompare('sparse100_10class_130_t6','sparse100_10class','_weight_hist_10Hz_comparison')

