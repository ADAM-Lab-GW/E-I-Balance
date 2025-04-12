# This script loads weight matrices from file
#  and then runs the network saving the simulation

from ast import parse
from contextlib import redirect_stderr
from surrogateParams import surrParams
import simLogger
import surrogateTraining
import weightAnalysis
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def makeMultiInputDataPlot(x_raw, y_raw, indices, tag, fileName='inputDataImage.png', dim=(3,5)):
    gs=GridSpec(*dim)
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        ax.imshow(x_raw[indices[i]].reshape(28,28), cmap=plt.cm.gray_r)
        #ax.axis("off")
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if(y_raw[indices[i]] == 0): color='red'
        else: color='blue'
        ax.spines['left'].set(color=color, linewidth=3)
        ax.spines['right'].set(color=color, linewidth=3)
        ax.spines['top'].set(color=color, linewidth=3)
        ax.spines['bottom'].set(color=color, linewidth=3)
    
    plt.suptitle(tag + ' input images', y=0.92, fontsize=18)
    plt.savefig(fileName)
    plt.close()

def makeMultiInputSpikePlot(x_spikes, y_spikes, indices, tag, fileName='inputDataSpikes.png', dim=(3,5)):
    gs=GridSpec(*dim)
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        #plt.scatter(x_spikes[indices[i]].to_dense().detach().cpu().numpy())
        xaxis = range(100)
        yaxis = range(784)
        plt.scatter(x_spikes[indices[i]]._indices()[0], x_spikes[indices[i]]._indices()[1], s=0.2, alpha=0.5)
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
        ax.set_ylim([784,0])
        ax.set_aspect(100.0/784.0)

    plt.suptitle(tag + ' input spikes', y=0.92, fontsize=18)
    plt.savefig(fileName)
    plt.close()   

parser = argparse.ArgumentParser(description='Training program for a particular learning rate')
parser.add_argument('-w1', '--w1_tag', help='Simtag for the input to hidden matrix pickle file', default='defaultSimtag')
parser.add_argument('-w2', '--w2_tag', help='Simtag for the hidden to output matrix pickle file', default='defaultSimtag')
parser.add_argument('-t', '--tag', help='Tag for images and file names', default='default')
parser.add_argument('-p', '--plot_dir', help='Directory for where the plots should be saved', default='.')
parser.add_argument('-o', '--obj_path', help='Folder path for the object directory containing the weight files', default='./data/obj')
args = parser.parse_args()

w1 = weightAnalysis.getWeightFromUniqueId(args.w1_tag, objFolder=args.obj_path)
w2 = weightAnalysis.getWeightFromUniqueId(args.w2_tag, objFolder=args.obj_path)

params = surrParams()
x_train, x_test, y_train, y_test = surrogateTraining.getDataSet()
x_batch, y_batch, batch_index = surrogateTraining.get_mini_batch(x_test, y_test, params, shuffle=False)
output, other_recordings = surrogateTraining.run_snn(x_batch.to_dense(), w1, w2, params)
mem_rec, spk_rec = other_recordings

makeMultiInputDataPlot(x_test, y_test, batch_index, args.tag, fileName=args.plot_dir+'/'+args.tag+'_inputImage.png')
makeMultiInputSpikePlot(x_batch, y_batch, batch_index, args.tag, fileName=args.plot_dir+'/'+args.tag+'_inputSpike.png')
surrogateTraining.plot_voltage_traces(mem=mem_rec, y_data=y_test, batch_indices=batch_index, spk=spk_rec, suptitle=args.tag+' hidden layer voltage', fileName=args.plot_dir+'/'+args.tag+'_hiddenVolt.png')
surrogateTraining.plot_voltage_traces(mem=output, y_data=y_test, batch_indices=batch_index, suptitle=args.tag+' output layer votlage', fileName=args.plot_dir+'/'+args.tag+'_outputVolt.png')