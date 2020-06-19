import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import TEP_Import as imp
from DataNode import Node
from Norm import Norm

class SFA(Node):
    '''
    Slow feature analysis class that takes a set of input signals and
    calculates the transformation matrix and slow features for the inputs
    '''
    # Status of SFA class
    
    # Enforced order:
    # Padding (optional) > Expansion (optional) > Training > Parting
    dynamic_copies = 0     # Number of padded copies (not including original) 
    expansion_order = 1    # Order of nonlinear expansion (1 -> no expansion)

    trained = False        # Has the model been trained yet
    parted = False         # Has the model been partitioned yet
    
    # Matrix dimensions of training data (m -> rows, N -> columns)
    delta = 1
    m_raw = None           # Input (raw) signals
    N_raw = None           # Input (raw) samples
    m = None               # Input (expanded and dynamized) signals
    N = None               # Input (expanded and dynamized) samples

    # Main members used for SFA equations
    # Note: If not expanded or dynamized then signals = signals_raw
    # s = Wx
    features = None        # Extracted features (s)
    trans_mat = None       # Transformation matrix (W)
    signals_raw = None     # Raw input signals
    NormObject = None      # Norm object used to normalize data
    signals = None         # Normalized, expanded, dynamized input signals (x)


    # Partitioning
    features_speed = None  # Speed of all features of training data
    Md = None              # Number of slowest features
    Me = None              # Number of fastest features (# of features - Md)
    slow_features = None   # The slowest (Md) features
    fast_features = None   # The fastest (Me) features

    # Test statistics
    T_d_crit = None        # Critical T^2 value for slowest features
    T_e_crit = None        # Critical T^2 value for fastest features 
    S_d_crit = None        # Critical S^2 value for slowest features
    S_e_crit = None        # Critical S^2 value for fastest features

    
    def __init__(self, signals_raw,dynamic_copies=None,expansion_order=None):
        '''
        Constructor method for SFA object
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

    def _preprocess_signals(self,signals_raw):
        '''
        Add lagged copies and perform nonlinear expansion
        '''
        signals = self.dynamize(signals_raw,self.dynamic_copies)
        signals = self.nonlinear_expansion(signals,self.expansion_order)
        return(signals)
    
    def _normalize_training(self):
        '''
        Creates the Norm object for future normalization and normalizes
        the training data
        '''
        NormObj = Norm(self.signals)
        signals_norm = NormObj.normalize()
        self.NormObj = NormObj
        self.signals_norm = signals_norm
        return
    
    def _transform_training(self):
        '''
        Transforms the whitened data into the features and stores 
        the transformation matrix and feature speeds
        '''
        Z = self.signals_norm

        # Approximate the first order time derivative of the signals
        Zdot = np.diff(Z)/self.delta
        
        # SVD of the var-cov matrix of Zdot
        Zdotcov = np.cov(Zdot)        
        PT, Omega, P = np.linalg.svd(Zdotcov,hermitian = True)

        # The loadings are ordered to find the slowest varying features
        self.features_speed, P = self._order(Omega,P)

        self.trans_mat = np.matmul(P,self.NormObj.white_mat)
        self.features = np.matmul(P,Z)
        return

    def _order(self,eigenvalues,eigenvectors):
        '''
        Orders a set of eigenvalue/eigenvector pairs by ascending eigenvalues
        '''
        p = eigenvalues.argsort()
        vals = eigenvalues[p]
        vecs = eigenvectors[p,:]
        
        return(vals,vecs)

    def train(self):
        '''
        Runs the SFA algorithm and returns the extracted features
        '''
        self._normalize_training()
        self._transform_training()
        self.trained = True
        return self.features

    def partition(self,q = 0.1):
        '''
        Partitions slow features into the slowest (1-q)*100% features
        and the fastest q*100% features
        '''
        if not self.trained:
            raise RuntimeError("Model has not been trained yet")

        X = self.signals_norm
        # Find slowness of input signals
        xdot = np.diff(X)/self.delta
        mdot = xdot.shape[0]
        Ndot = xdot.shape[1]
        sig_speed = np.zeros(mdot)
        for i in range(mdot):
            sig_speed[i] = np.matmul(xdot[i,:],xdot[i,:].T)/(Ndot-1)

        # Find where to partition slow features
        threshold = np.quantile(sig_speed,1-q,interpolation='lower')
        sf_speed = self.features_speed
        for i in range(sf_speed.size):
            if sf_speed[i] > threshold:
                self.Me = i
                self.Md = sf_speed.size-i
                break

        
        self.slow_features = np.copy(self.features[:self.Me,:])
        self.slow_features = np.copy(self.features[self.Me:,:])
        self.parted = True
        self.calculate_crit_values()
        return

    def partition_manual(self,Me):
        '''
        Manually set how many slow features you want to extract where
        Me is the number of slow features
        '''
        self.Me = Me
        self.Md = self.features_speed.size - Me
        self.slow_features = np.copy(self.features[:self.Me,:])
        self.fast_features = np.copy(self.features[self.Me:,:])
        self.parted = True
        self.calculate_crit_values()
        return
    
    def calculate_crit_values(self,alpha=0.01):
        '''
        Calculate critical values for monitoring
        '''
        if alpha > 1:
            raise ValueError("Confidence level is capped at 1")
        p = 1 - alpha
        N  = self.N
        Md = self.Md
        Me = self.Me
        gd = (Md*(N**2-2*N))/((N-1)*(N-Md-1))
        ge = (Me*(N**2-2*N))/((N-1)*(N-Me-1))

        self.T_d_crit = stats.chi2.ppf(p,Md)
        self.T_e_crit = stats.chi2.ppf(p,Me)
        self.S_d_crit = gd*stats.f.ppf(p,Md,N-Md-1)
        self.S_e_crit = ge*stats.f.ppf(p,Me,N-Me-1)

        return(self.T_d_crit,self.T_e_crit,self.S_d_crit,self.S_e_crit)


    def calculate_monitors(self, online_data):
        # Takes in a batch of online data and returns an array
        # that contains the statistical indices
        # Data should have enough samples to be padded as previously done
        if not self.parted:
            raise RuntimeError("Signals have not been partitioned yet")

        # Add lagged copies and perform nonlinear expansion
        online_data = self.dynamize(online_data,self.dynamic_copies)
        online_data = self.nonlinear_expansion(online_data,self.expansion_order)
        
        X = self.NormObj.center_similar(online_data)
        W = self.trans_mat
        features_online = np.matmul(W,X)
        
        # Split slow features into fastest and slowest based on training data
        s_d = features_online[:self.Me,:]
        s_e = features_online[self.Me:,:]
        # Calculate time derivatives
        sdot_d = np.diff(s_d)/self.delta
        sdot_e = np.diff(s_e)/self.delta

        N = s_d.shape[1]
        N_dot = sdot_d.shape[1]

        T_squared_d = np.zeros((N))
        T_squared_e = np.zeros((N))
        S_squared_d = np.zeros((N_dot))
        S_squared_e = np.zeros((N_dot))

        for i in range(N):
            sample_d = s_d[:,i]
            sample_e = s_e[:,i]
            T_squared_d[i] = np.matmul(sample_d.T,sample_d)
            T_squared_e[i] = np.matmul(sample_e.T,sample_e)


        Omega = self.features_speed
        Omega_d = Omega[:self.Me]
        Omega_e = Omega[self.Me:]
        
        Omega_d_inv = np.diag(Omega_d**(-1))
        Omega_e_inv = np.diag(Omega_e**(-1))
        
        for i in range(N_dot):
            sample_d = sdot_d[:,i]
            sample_e = sdot_e[:,i]
            S_squared_d[i] = np.matmul(np.matmul
                                       (sample_d.T,Omega_d_inv),sample_d)
            S_squared_e[i] = np.matmul(np.matmul
                                       (sample_e.T,Omega_e_inv),sample_e)

        return(T_squared_d,T_squared_e,S_squared_d,S_squared_e,s_d)


if __name__ == "__main__":    
    # Import TEP and set up input data
    training_sets = list(imp.importTrainingSets([0]))
    testing_sets  = list(imp.importTestSets([4,5,10]))

    X   = training_sets[0][1]
    T4  = testing_sets[0][1].T
    T5  = testing_sets[1][1].T
    T10 = testing_sets[2][1].T
    
    # Delete ignored variables
    ignored_var = list(range(22,41))
    X = np.delete(X,ignored_var,axis=0)
    T4 = np.delete(T4,ignored_var,axis=0)
    T5 = np.delete(T5,ignored_var,axis=0)
    T10 = np.delete(T10,ignored_var,axis=0)

    d = 2    # Lagged copies
    q = 0.1  # Partition fraction
    n = 1    # Expansion order
    Me = 55  # Slow features to keep
    
    # Run SFA
    SlowFeature = SFA(X,d,n)
    SlowFeature.delta = 3
    Y = SlowFeature.train()
    SlowFeature.partition(q)
    SlowFeature.partition_manual(Me)
    
    T_dc, T_ec, S_dc, S_ec = SlowFeature.calculate_crit_values()

    data_iterable = [("Orig",X),("IDV(4)",T4),("IDV(5)",T5),("IDV(10)",T10)]
    num_data = len(data_iterable)
    
    monitors = plt.figure("Monitors")
    plt.subplots_adjust(wspace=0.4)    
    col_pos = 1
    
    for name, test in data_iterable:
        T_d, T_e, S_d, S_e, SF  = SlowFeature.calculate_monitors(test)
        threshold = np.ones(test.shape[1])
        
        # Plotting
        plt.figure("Monitors")
        plt.subplot(4,num_data,col_pos)
        plt.title(name)
        plt.plot(T_d)
        plt.plot(T_dc*threshold)
        plt.ylabel("$T^2$")
        plt.xticks([])

        plt.subplot(4,num_data,col_pos+num_data)
        plt.plot(T_e)
        plt.plot(T_ec*threshold)
        plt.ylabel("$T^2_e$")
        plt.xticks([])

        plt.subplot(4,num_data,col_pos+num_data*2)
        plt.plot(S_d)
        plt.plot(S_dc*threshold)
        plt.ylabel("$S^2$")
        plt.xticks([])

        plt.subplot(4,num_data,col_pos+num_data*3)
        plt.plot(S_e)
        plt.plot(S_ec*threshold)
        plt.ylabel("$S^2_e$")
        plt.xlabel("Sample")
        
        col_pos += 1


    mid = int(Y.shape[0]/2)
    slowest = Y[:5,:]
    middle = Y[mid:mid+5,:]
    fastest = Y[-5:,:]
    plt.figure("Slow Features")
    for i in range(5):
        plt.subplot(5,3,3*i+1)
        plt.plot(slowest[i,:])
        if i == 0:
            plt.title("Slowest")
            
        plt.subplot(5,3,3*i+2)
        plt.plot(middle[i,:])
        if i == 0:
            plt.title("Middle")
            
        plt.subplot(5,3,3*i+3)
        plt.plot(fastest[i,:])
        if i == 0:
            plt.title("Fastest")
            
    plt.show()
