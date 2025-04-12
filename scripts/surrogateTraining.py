import os
#from matplotlib import test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
#import torchvision
import pickle

from surrogateParams import surrParams
import simLogger

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

def getDataSet(params):
    # we are going to load the datasets pre processed
    #  from pickled files. See the dataset folder for
    #  scripts to download the datasets
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    
    if(params.data_set == 'FashionMNIST'):
        # load fashion MNIST dataset
        with open('./dataset/FashionMNIST/FashionMNIST_x_training.pkl','rb') as f:
            x_train = pickle.load(f)

        with open('./dataset/FashionMNIST/FashionMNIST_x_test.pkl','rb') as f:
            x_test = pickle.load(f)

        with open('./dataset/FashionMNIST/FashionMNIST_y_training.pkl','rb') as f:
            y_train = pickle.load(f)
        
        with open('./dataset/FashionMNIST/FashionMNIST_y_test.pkl','rb') as f:
            y_test = pickle.load(f)

    elif(params.data_set == 'SHD'):
        # load the spiking heidelberg digits dataset
        with open('./dataset/SHD/SHD_x_training.pkl','rb') as f:
            x_train = pickle.load(f)

        with open('./dataset/SHD/SHD_x_test.pkl','rb') as f:
            x_test = pickle.load(f)

        with open('./dataset/SHD/SHD_y_training.pkl','rb') as f:
            y_train = pickle.load(f)
        
        with open('./dataset/SHD/SHD_y_test.pkl','rb') as f:
            y_test = pickle.load(f)
    else:
        print("No known dataset selected")
        exit()

    train_idx = (y_train < params.nb_outputs)
    test_idx = (y_test < params.nb_outputs)
    
    x_train = x_train.to_dense()[train_idx].to_sparse()
    x_test = x_test.to_dense()[test_idx].to_sparse()
    y_train = y_train[train_idx]
    y_test = y_test[test_idx]

    return x_train, x_test, y_train, y_test

def sparse_data_generator_preloaded(X,y,params,shuffle=True):
    X = X.to_dense()
    labels_ = np.array(y,dtype=int)
    number_of_batches = len(X)//params.batch_size
    sample_index = np.arange(len(X))
    
    if shuffle:
        np.random.shuffle(sample_index)
        
    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[params.batch_size*counter:params.batch_size*(counter+1)]
        
        X_batch = X[batch_index]
        y_batch = y[batch_index]
        
        yield (X_batch.to_sparse().to(device=device), y_batch.to(device=device), batch_index)
        
        counter += 1

def initWeights(params):
    weight_scale = 7*(1.0- params.beta) # this should give us some spikes to begin with
    
    w1 = torch.empty((params.nb_inputs, params.nb_hidden),  device=device, dtype=dtype, requires_grad=True)
    if(params.sparse_w1 == False):
        torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(params.nb_inputs))
        with torch.no_grad():
            w1 = torch.abs(w1)
        w1.requires_grad = True
    else:
        w1init = w1
        torch.nn.init.sparse_(w1init, sparsity=params.sparsity_w1, std=params.std_w1)
        with torch.no_grad():
            w1 = torch.abs(w1init)
        w1.requires_grad = True
    

    w2 = torch.empty((params.nb_hidden, params.nb_outputs), device=device, dtype=dtype, requires_grad=True)
    if(params.sparse_w2 == False):
        torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(params.nb_hidden))
        with torch.no_grad():
            if(params.isIZ):
                w2[params.isExc,:] = torch.abs(w2[params.isExc,:])
                w2[~params.isExc,:] = -1*torch.abs(w2[~params.isExc,:])
            else :
                w2[0:int(params.nb_hidden*params.excHidRat),:] = torch.abs(w2[0:int(params.nb_hidden*params.excHidRat),:])
                w2[int(params.nb_hidden*params.excHidRat):,:] = -1*torch.abs(w2[int(params.nb_hidden*params.excHidRat):,:])
        w2.requires_grad = True
    else:
        w2initClone = w2.clone()
        w2init = w2
        torch.nn.init.sparse_(w2init, sparsity=params.sparsity_w2, std=params.std_w2)
        with torch.no_grad():
            if(params.isIZ):
                w2initClone[params.isExc,:] = torch.abs(w2init[params.isExc,:])
                w2initClone[~params.isExc,:] = -1*torch.abs(w2init[~params.isExc,:])
            else :
                w2initClone[0:int(params.nb_hidden*params.excHidRat),:] = torch.abs(w2init[0:int(params.nb_hidden*params.excHidRat),:])
                w2initClone[int(params.nb_hidden*params.excHidRat):,:] = -1*torch.abs(w2init[int(params.nb_hidden*params.excHidRat):,:])
        w2 = w2initClone.clone().detach()
        w2.requires_grad = True

    print("init done")
    return w1, w2

def plot_voltage_traces(mem, suptitle='', fileName='volt.png', spk=None, dim=(3,5), y_data=None, batch_indices=None, spike_height=5, titleSize=18):
    gs=GridSpec(*dim)
    if spk is not None:
        dat = 1.0*mem
        dat[spk>0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)

        if (y_data is None):
            ax.plot(dat[i])
            ax.axis("off")
        else:
            if(y_data[batch_indices[i]] == 0): color='red'
            else: color='blue'
            ax.spines['left'].set(color=color, linewidth=3)
            ax.spines['right'].set(color=color, linewidth=3)
            ax.spines['top'].set(color=color, linewidth=3)
            ax.spines['bottom'].set(color=color, linewidth=3)
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            ax.plot(dat[i])

    plt.suptitle(suptitle, y=0.95, fontsize=titleSize)
    plt.savefig(fileName)
    plt.close()


def run_snn(inputs, w1, w2, params):
    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    syn = torch.zeros((params.batch_size,params.nb_hidden), device=device, dtype=dtype)
    
    if(not params.isIZ):
        mem = torch.ones((params.batch_size,params.nb_hidden), device=device, dtype=dtype)*params.vRest
    else:
        mem = torch.ones((params.batch_size,params.nb_hidden), device=device, dtype=dtype)*params.vRest
        u = torch.zeros((params.batch_size,params.nb_hidden), device=device, dtype=dtype)
        
    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    for t in range(params.nb_steps):

        if(not params.isIZ):
            mthr = mem-params.vPeak
            out = params.spike_fn(mthr)
            rst = out.detach() # We do not want to backprop through the reset

            new_syn = params.alpha*syn +h1[:,t]
            new_mem = (params.beta*(mem-params.vRest)+params.vRest+syn)*(1.0-rst)+(rst*params.vRest)

        else: 
            out = params.spike_fn(mem)
            rst = out.detach() # We do not want to backprop through the reset

            new_syn = params.alpha*syn +h1[:,t]
            new_mem = (mem + 1000*params.time_step*(1/params.C *(params.k*(mem-params.vRest)*(mem-params.vThresh) - u + 1000*syn))) * (1.0-rst) + (rst * params.vMin)
            new_u = (u + 1000*params.time_step*(params.a*(params.b*(mem-params.vRest) - u))) * (1.0-rst) + (rst * (u+params.d))
            u = new_u
            
        mem_rec.append(mem)
        spk_rec.append(out)
        
        mem = new_mem
        syn = new_syn
    #if(params.debug):   
        #simLogger.saveObj(params.simTag,'mem_rec',mem_rec)
    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)  

    # Readout layer
    h2= torch.einsum("abc,cd->abd", (spk_rec, w2))
    flt = torch.zeros((params.batch_size,params.nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((params.batch_size,params.nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(params.nb_steps):
        new_flt = params.alpha*flt +h2[:,t]
        new_out = params.beta*out +flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec,dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs

def boundWeights(w1, w2, params):
    w1new = torch.nn.functional.relu(w1)
    w1 = w1new.clone().detach()
    w1.requires_grad = True
    
    w2new = w2.clone()
    if(params.isIZ):
        w2new[params.isExc, :] = torch.nn.functional.relu(w2[params.isExc,:])
        w2new[~params.isExc, :] = -1*torch.nn.functional.relu(-1*w2[~params.isExc,:])
    else:
        w2new[0:int(params.nb_hidden*params.excHidRat),:] = torch.nn.functional.relu(w2[0:int(params.nb_hidden*params.excHidRat),:])
        w2new[int(params.nb_hidden*params.excHidRat):,:] = -1*torch.nn.functional.relu(-1*w2[int(params.nb_hidden*params.excHidRat):,:])
    
    w2 = w2new.clone().detach()
    w2.requires_grad = True

    return (w1,w2)

def compute_classification_accuracy(x_data, y_data, params, w1, w2):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    for x_local, y_local, batch_index in sparse_data_generator_preloaded(x_data, y_data, params, shuffle=False):
        output,_ = run_snn(x_local.to_dense(), w1, w2, params)
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)

def train(x_data, y_data, x_test, y_test, w1, w2, params):
    if(params.start_epoch == None or params.start_epoch == -1):
        simLogger.saveObj(params.simTag, 'w1_init', w1, makeNote=True)
        simLogger.saveObj(params.simTag, 'w2_init', w2, makeNote=True)

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    acc_hist = []

    acc = compute_classification_accuracy(x_test, y_test, params, w1, w2)
    acc_hist.append(acc)
    
    epochLst = range(params.nb_epochs)
    if(params.start_epoch != None):
        epochLst = range(params.start_epoch, params.start_epoch+params.nb_epochs)

    for e in epochLst:
        batch_num=0
        local_loss = []
        for x_local, y_local, batch_index in sparse_data_generator_preloaded(x_data, y_data, params):
            optParams = [w1,w2]
            optimizer = torch.optim.SGD(optParams, lr=params.lr)
            print("BATCH : " + str(batch_num))
            output,other_rec = run_snn(x_local.to_dense(), w1, w2, params)
            [mem_rec,spk_rec] = other_rec
            #if(params.debug):
                #simLogger.saveObj(params.simTag, 'mem_rec_e'+str(e)+'_b'+str(batch_num), mem_rec, makeNote=True)
                #simLogger.saveObj(params.simTag, 'input_e'+str(e)+'_b'+str(batch_num), x_local.to_dense(), makeNote=True)
                #simLogger.saveObj(params.simTag, 'spk_rec_e'+str(e)+'_b'+str(batch_num), spk_rec, makeNote=True)
                #simLogger.saveObj(params.simTag, 'output_e'+str(e)+'_b'+str(batch_num), output, makeNote=True)

            m,_=torch.max(output,1)
            log_p_y = log_softmax_fn(m)
            loss_val = loss_fn(log_p_y, y_local.type(torch.LongTensor))
            
            #if(params.debug):
                #simLogger.saveObj(params.simTag, 'loss_val_e'+str(e)+'_b'+str(batch_num), loss_val, makeNote=True)
                #simLogger.saveObj(params.simTag, 'loss_p_y_e'+str(e)+'_b'+str(batch_num), loss_p_y, makeNote=True)

            optimizer.zero_grad()
            w1.retain_grad()
            w2.retain_grad()
            loss_val.backward()

            if(params.debug):
                grad = w1.grad
                simLogger.logNotes('GRAD: '+str(grad[0,0]))
                simLogger.saveObj(params.simTag, 'w1_grad_e'+str(e)+'_b'+str(batch_num), grad, makeNote=True)
            
                simLogger.saveObj(params.simTag, 'w1_pre_update_e'+str(e)+'_b'+str(batch_num), w1, makeNote=True)
                simLogger.saveObj(params.simTag, 'w2_pre_update_e'+str(e)+'_b'+str(batch_num), w2, makeNote=True)
            optimizer.step()
            local_loss.append(loss_val.item())
            if(params.isNoisy):
                if(params.debug):
                    simLogger.saveObj(params.simTag, 'w1_pre_noise_e'+str(e)+'_b'+str(batch_num), w1, makeNote=True)
                    simLogger.saveObj(params.simTag, 'w2_pre_noise_e'+str(e)+'_b'+str(batch_num), w2, makeNote=True)
                w1Noise = torch.randn(params.nb_inputs, params.nb_hidden)*params.w1NoiseSTD
                w2Noise = torch.randn(params.nb_hidden, params.nb_outputs)*params.w2NoiseSTD
                w1 = w1 + w1Noise
                w2 = w2 + w2Noise
                if(params.debug):
                    simLogger.saveObj(params.simTag, 'w1_noise_e'+str(e)+'_b'+str(batch_num), w1Noise, makeNote=True)
                    simLogger.saveObj(params.simTag, 'w2_noise_e'+str(e)+'_b'+str(batch_num), w2Noise, makeNote=True)
            
            (w1, w2) = boundWeights(w1, w2, params)
            if(params.debug):
                simLogger.saveObj(params.simTag, 'w1_post_noise_e'+str(e)+'_b'+str(batch_num), w1, makeNote=True)
                simLogger.saveObj(params.simTag, 'w2_post_noise_e'+str(e)+'_b'+str(batch_num), w2, makeNote=True)
            
            batch_num+=1
        
        simLogger.saveObj(params.simTag, 'w1_e'+str(e), w1, makeNote=True)
        simLogger.saveObj(params.simTag, 'w2_e'+str(e), w2, makeNote=True)
        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5f"%(e+1,mean_loss))
        loss_hist.append(mean_loss)

        acc = compute_classification_accuracy(x_test, y_test, params, w1, w2)
        print("Epoch %i: acc=%.5f"%(e+1,acc))
        acc_hist.append(acc)

        #if mean_loss>10:
        #    break
        
    return loss_hist, acc_hist
     
def get_mini_batch(x_data, y_data, params, shuffle=False):
    miniData = None
    for x_batch, y_batch, batch_index in sparse_data_generator_preloaded(x_data, y_data, params, shuffle=shuffle):
        miniData = (x_batch, y_batch, batch_index)
        break

    return miniData

def main(params = surrParams(), w1=None, w2=None):
    # collect data (training and testing)
    x_train, x_test, y_train, y_test = getDataSet(params)

    # initialize the weights
    if(w1==None and w2==None):
        w1, w2 = initWeights(params)
        (w1, w2) = boundWeights(w1, w2, params)

    loss_hist,acc_hist = train(x_train, y_train, x_test, y_test, w1, w2, params)
    print("ACCURACY: " + str(max(acc_hist)))
    return acc_hist, loss_hist

if __name__ == "__main__":
    acc_hist, loss_hist = main()
    print("ACCURACY: " + str(acc_hist))
