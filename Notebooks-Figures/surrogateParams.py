import numpy as np

class surrParams:
    def __init__(self):
        self.version = 1.0
        self.simTag = None

        self.nb_inputs  = 28*28
        self.nb_hidden  = 100
        self.excHidRat = 0.5
        self.nb_outputs = 10
        self.time_step = 1e-3
        self.nb_steps  = 100
        self.batch_size = 256
        self.nb_epochs = 30
        self.lr = 2e-3
        self.tau_mem = 10e-3
        self.tau_syn = 5e-3
        self.alpha   = float(np.exp(-self.time_step/self.tau_syn))
        self.beta    = float(np.exp(-self.time_step/self.tau_mem))

        self.sparse_w1 = False
        self.sparse_w2 = False
        self.sparsity_w1 = 0.0
        self.sparsity_w2 = 0.0
        self.std_w1 = 0.0
        self.std_w2 = 0.0
