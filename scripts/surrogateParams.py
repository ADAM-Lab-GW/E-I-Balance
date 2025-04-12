import numpy as np
import torch
from SurrGradSpikeLIF import SurrGradSpikeLIF
from SurrGradSpikeIZ import SurrGradSpikeIZ

class surrParams:
    def __init__(self):
        self.version = 1.2
        self.simTag = "Test"
        self.debug = False

        # the other option for dataset is the "SHD"
        self.data_set = "FashionMNIST"

        self.nb_inputs  = 28*28
        self.nb_hidden  = 100
        self.excHidRat = 0.5
        self.nb_outputs = 10
        self.time_step = 1e-3
        self.nb_steps  = 100
        self.batch_size = 256
        self.nb_epochs = 30
        self.start_epoch = None
        self.lr = 2e-3
        self.tau_mem = 10e-3
        self.tau_syn = 5e-3
        self.vRest = 0.0
        self.vPeak = 1.0
        self.alpha   = float(np.exp(-self.time_step/self.tau_syn))
        self.beta    = float(np.exp(-self.time_step/self.tau_mem))

        self.sparse_w1 = False
        self.sparse_w2 = False
        self.sparsity_w1 = 0.0
        self.sparsity_w2 = 0.0
        self.std_w1 = 0.0
        self.std_w2 = 0.0

        self.spike_fn  = SurrGradSpikeLIF.apply

        # IZ params
        self.isIZ = False
        self.C = None     
        self.k = None
        self.vMin = None
        self.vThresh = None
        self.a = None
        self.b = None
        self.d = None
        self.isExc = None

        # Noisy weight updates
        self.isNoisy = False # default no noise
        self.w1NoiseSTD = 0.0
        self.w2NoiseSTD = 0.0

    def setLIFParams(self,vRest,vPeak,tau_mem):
        # if the LIF params are updated we will need this method
        #  so that the appropriate surrogate gradient can also
        #  be calculated
        self.vRest = vRest
        self.vPeak = vPeak
        self.tau_mem = tau_mem
        self.beta    = float(np.exp(-self.time_step/tau_mem))
        SurrGradSpikeLIF.setVLevels(vPeak, vRest)
        self.spike_fn = SurrGradSpikeLIF.apply

    def setIZParams(self,C,k,vMin,vRest,vThresh,vPeak,a,b,d,isExc,nIdMin,nIdMax):
        # set the iz params in the IZ matrix to the parameters listed
        # this is set for only the neuron indices in the range [nIdMin,nIdMax)
        #  (including the lower bound not including the upper bound)
        # if the torch vectors for each variable have not been setup, they will
        #  be initialized with all 0's and then filled in where appropriate
        # note that if the batch size or hidden layer size changes, this will
        #  need to be rerun to update the matrix sizes

        if(not self.isIZ):
            self.isIZ = True
            self.C = torch.zeros(self.batch_size,self.nb_hidden)
            self.k = torch.zeros(self.batch_size,self.nb_hidden)
            self.vMin = torch.zeros(self.batch_size,self.nb_hidden)
            self.vRest = torch.zeros(self.batch_size,self.nb_hidden)
            self.vThresh = torch.zeros(self.batch_size,self.nb_hidden)
            self.vPeak = torch.zeros(self.batch_size,self.nb_hidden)
            self.a = torch.zeros(self.batch_size,self.nb_hidden)
            self.b = torch.zeros(self.batch_size,self.nb_hidden)
            self.d = torch.zeros(self.batch_size,self.nb_hidden)
            self.isExc = torch.zeros(self.nb_hidden, dtype=torch.bool)
        
        self.C[:,nIdMin:nIdMax] = C
        self.k[:,nIdMin:nIdMax] = k
        self.vMin[:,nIdMin:nIdMax] = vMin
        self.vRest[:,nIdMin:nIdMax] = vRest
        self.vThresh[:,nIdMin:nIdMax] = vThresh
        self.vPeak[:,nIdMin:nIdMax] = vPeak
        self.a[:,nIdMin:nIdMax] = a
        self.b[:,nIdMin:nIdMax] = b
        self.d[:,nIdMin:nIdMax] = d
        self.isExc[nIdMin:nIdMax] = isExc

        SurrGradSpikeIZ.setVPeak(self.vPeak)
        SurrGradSpikeIZ.setVThresh(self.vThresh)
        self.spike_fn = SurrGradSpikeIZ.apply

