import statistics
import surrogateTraining
from surrogateParams import surrParams
import weightAnalysis
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def makeMultiHiddenSpikePlot(x_spikes, y_spikes, indices, tag, fileName = 'hiddenDataSpikes.png', plotdir='.', dim=(3,5)):
    statistics = []
    gs=GridSpec(*dim)
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        #plt.scatter(x_spikes[indices[i]].to_dense().detach().cpu().numpy())
        xaxis = range(100)
        yaxis = range(100)
        addSpk = x_spikes[indices[i]].coalesce().values().detach().numpy()>0
        
        plt.scatter(x_spikes[indices[i]]._indices()[0][addSpk], x_spikes[indices[i]]._indices()[1][addSpk], c=x_spikes[indices[i]].coalesce().values().detach().numpy()[addSpk], s=0.2, alpha=0.5)
        #ax.axis("off")
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if(y_spikes[indices[i]] == 0): color='red'
        else: color='blue'
        ax.spines['left'].set(color=color, linewidth=3)
        ax.spines['right'].set(color=color, linewidth=3)
        ax.spines['top'].set(color=color, linewidth=3)
        ax.spines['bottom'].set(color=color, linewidth=3)
        ax.set_xlim([0,100])
        ax.set_ylim([0,100])
        ax.set_aspect(1)

    plt.suptitle(tag + ' added hidden spikes', y=0.92, fontsize=18)
    plt.savefig(plotdir+'/'+'added_'+fileName)
    plt.close() 

    gs=GridSpec(*dim)
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        #plt.scatter(x_spikes[indices[i]].to_dense().detach().cpu().numpy())
        xaxis = range(100)
        yaxis = range(100)
        removeSpk = x_spikes[indices[i]].coalesce().values().detach().numpy()<0
        
        plt.scatter(x_spikes[indices[i]]._indices()[0][removeSpk], x_spikes[indices[i]]._indices()[1][removeSpk], c=x_spikes[indices[i]].coalesce().values().detach().numpy()[removeSpk], s=0.2, alpha=0.5)
        #ax.axis("off")
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if(y_spikes[indices[i]] == 0): color='red'
        else: color='blue'
        ax.spines['left'].set(color=color, linewidth=3)
        ax.spines['right'].set(color=color, linewidth=3)
        ax.spines['top'].set(color=color, linewidth=3)
        ax.spines['bottom'].set(color=color, linewidth=3)
        ax.set_xlim([0,100])
        ax.set_ylim([0,100])
        ax.set_aspect(1)

    plt.suptitle(tag + ' removed hidden spikes', y=0.92, fontsize=18)
    plt.savefig(plotdir+'/'+'removed_'+fileName)
    plt.close() 
    

def compareNetworks(w1A, w2A, w1B, w2B, tag='', plotdir='.'):
    # comparing networks A and B

    params = surrParams()
    x_train, x_test, y_train, y_test = surrogateTraining.getDataSet()
    x_batch, y_batch, batch_index = surrogateTraining.get_mini_batch(x_test, y_test, params, shuffle=False)
    outputA, other_recordingsA = surrogateTraining.run_snn(x_batch.to_dense(), w1A, w2A, params)
    mem_recA, spk_recA = other_recordingsA
    outputB, other_recordingsB = surrogateTraining.run_snn(x_batch.to_dense(), w1B, w2B, params)
    mem_recB, spk_recB = other_recordingsB

    outputDiff = outputB-outputA
    spkDiff = spk_recB - spk_recA
    #memDiff = mem_recB

    statistics = []
    for i in range(15):
        countChange = np.sum(spkDiff.to_sparse()[batch_index[i]].coalesce().values().detach().numpy())
        addedSpk = np.count_nonzero(spkDiff.to_sparse()[batch_index[i]].coalesce().values().detach().numpy() > 0)
        removedSpk = np.count_nonzero(spkDiff.to_sparse()[batch_index[i]].coalesce().values().detach().numpy() < 0)
        statistics.append([countChange, addedSpk, removedSpk])

    surrogateTraining.plot_voltage_traces(mem=outputDiff, y_data=y_test, batch_indices=batch_index, titleSize=14, suptitle=tag+' Difference in output layer voltage after training', fileName=plotdir+'/'+tag+'_diffOutputVolt.png')
    makeMultiHiddenSpikePlot(spkDiff.to_sparse(), y_batch, batch_index, tag, fileName = tag+'_hiddenSpikes.png', plotdir='./spikePlots', dim=(3,5))
    return outputDiff, spkDiff, statistics

for t in range(10):
    w1A = weightAnalysis.getWeightFromUniqueId('PEG_lr8_t'+str(t)+'_w1_init', objFolder='./PEG_lr8/PEG_lr8')
    w2A = weightAnalysis.getWeightFromUniqueId('PEG_lr8_t'+str(t)+'_w2_init', objFolder='./PEG_lr8/PEG_lr8')
    w1B = weightAnalysis.getWeightFromUniqueId('PEG_lr8_t'+str(t)+'_w1_e29_b45', objFolder='./PEG_lr8/PEG_lr8')
    w2B = weightAnalysis.getWeightFromUniqueId('PEG_lr8_t'+str(t)+'_w2_e29_b45', objFolder='./PEG_lr8/PEG_lr8')

    outputDiff, spkDiff, statistics = compareNetworks(w1A, w2A, w1B, w2B, tag='PEG_lr8_t'+str(t), plotdir='./voltagePlots')
    print('TRIAL '+ str(t))
    print(statistics)

print(spkDiff.size())
print(spkDiff.to_sparse())