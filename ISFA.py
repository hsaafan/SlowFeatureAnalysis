import numpy as np

from SFAClass import SFA
from WeightedNorm import WeightedNorm

class ISFA(SFA)
    '''
    Iterative Sslow feature analysis class that takes a set of input signals
    and calculates the slow features for the inputs
    '''
    weights = None
    grouped_variables = None
    def __init__(self, signals_raw, grouped_variables, weights
                 dynamic_copies = None, expansion_order = None):
        '''
        Constructor method for ISFA object
        Takes an m by N numpy ndarray input where m is the number of 
        signals and N is the number of samples. 
        Returns an SFA object.
        '''
        # Store original signals
        self.signals_raw = signals_raw
        self.m_raw = signals_raw.shape[0]
        self.N_raw = signals_raw.shape[1]
        
        # Store other inputs
        self.dynamic_copies = dynamic_copies
        self.expansion_order = expansion_order

        # Store dynamized and expanded signals
        X = self._preprocess_signals(signals_raw)
        self.m = X.shape[0]
        self.N = X.shape[1]
        self.signals = X

        return

    def _calculate_AB(self):
        
        return

    def _calculate_trans_mat(self):

        return

    def _calculate_sample_weight(self):

        return

    def _iterate(self):

        return
