import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import warnings
from itertools import combinations_with_replacement as combinations
from math import factorial as fac
from math import floor

import TEP_Import as imp

class SFA:
    '''
    Slow feature analysis class that takes a set of input signals and
    calculates the transformation matrix and slow features for the inputs
    '''
    # Status of SFA class
    # TODO: Check if order of padding and expansion matters
    # Enforced order:
    # Padding (optional) > Expansion (optional) > Training > Parting
    padded = False         # Has x been padded with time-delayed copies
    padded_copies = 0      # Number of padded copies (not including original) 
    expanded = False       # Has there been non-linear expansion on x
    expanded_order = 1     # Order of nonlinear expansion
    trained = False        # Has the model been trained yet
    parted = False         # Has the model been partitioned yet
    
    
    # Matrix dimensions of training data (m -> rows, N -> columns)
    m = None               # Input signals
    N = None               # Input samples
    m_dyn = None           # Signals available after padding (m_dyn > m)
    N_dyn = None           # Samples available after padding (N_dyn < N)
    m_exp = None           # Signals available after expansion (m_exp > m)
    # Expansion doesn't affect number of available samples

    # Main members used for SFA equations
    # s = Wx
    # If not padded:   signals_dyn = signals
    # If not expanded: signals_exp = signals_dyn
    # Use signals_exp for rest of algorithm
    slow_features = None   # Slow features (s)
    trans_mat = None       # Transformation matrix (W)
    signals = None         # Raw input signals
    signals_dyn = None     # Padded raw input signals
    signals_exp = None     # Expanded input signals
    offset = None          # The mean of the training signals
    signals_cent = None    # Signals centered around 0 (x)
    # z = Qx
    signals_norm = None    # Normalized signals (z)
    white_mat = None       # Whitening matrix (Q)
    # s = Pz
    trans_nor_mat = None   # Transformation matrix for normalized signals (P)


    # Partitioning and test statistics
    slow_features_speed = None
    Md = None              # Number of slowest features
    Me = None              # Number of fastest features
    gd = None              # Coefficient used for slower S^2 distribution
    ge = None              # Coefficient used for faster S^2 distribution/
    slow_features_d = None # The slowest features
    slow_features_e = None # The fastest features
    T_d_crit = None        # Critical T^2 value for slowest features
    T_e_crit = None        # Critical T^2 value for fastest features 
    S_d_crit = None        # Critical S^2 value for slowest features
    S_e_crit = None        # Critical S^2 value for fastest features

    def __init__(self, X):
        '''
        Constructor method for SFA object
        Takes an m by N numpy ndarray input where m is the number of 
        signals and N is the number of samples. 
        Returns an SFA object.
        '''
        if not isinstance(X, np.ndarray):
            raise TypeError("Expected a numpy ndarray object")
            
        if X.ndim == 1:
            # For 1-D arrays, assume there is one signal with multiple samples
            self.m = 1
            self.N = X.shape[0]
        elif X.ndim == 2:
            self.m = X.shape[0]
            self.N = X.shape[1]
        else:
            raise TypeError("Array input is not one or two dimensional")

        if self.m > self.N:
            warnings.warn("There are more signals than samples: " +
                          "Check that the data has been entered correctly",
                          RuntimeWarning)

        self.signals = X

        return

    def __dynamize_training(self, copies):
        '''
        
        '''
        signals = self.signals
        self.m_dyn = self.m*(copies+1)
        self.N_dyn = self.N - copies

        X = np.zeros((self.m_dyn,self.N_dyn))

        for i in range(0,copies+1):
            start_row = i * self.m
            stop_row  = (i + 1) *self. m
            start_col = i
            stop_col = self.N - copies + i
            X[start_row:stop_row,:] = signals[:,start_col:stop_col]
        
        self.signals_dyn = X
        self.padded = True
        self.padded_copies = copies

        return
    
    
    def __expand_training(self,n=2):
        '''
        Performs nonlinear expansion on signals where n is the
        order of expanson performed
        '''
        if self.padded:
            num_signals = self.m_dyn
            num_samples = self.N_dyn
            X = self.signals_dyn
        else:
            num_signals = self.m
            num_samples = self.N
            X = self.signals

        # Find dimensions of expanded matrix
        k = num_signals # Order 1
        for r in range(2,n+1):
            # Order 2 -> n
            k += fac(r+num_signals-1)/(fac(r)*(fac(num_signals-1)))
        k = int(k)

        X_exp = np.empty((k,num_samples))
        

        # Add expanded signals
        # Order 1
        X_exp[0:num_signals,:] = X
        
        pos = num_signals # Where to add new signal
        for order in range(2,n+1):
            # Order 2 -> n
            for comb in combinations(range(num_signals),order):
                exp_signal = np.ones((1,num_samples))
                for i in comb:
                    exp_signal = exp_signal*X[i,:]
                X_exp[pos,:] = exp_signal
                pos += 1
        
        self.m_exp = k
        self.signals_exp = X_exp
        return

    def __center_training(self):
        '''
        Private method used to center the training data and store
        the offset for centering online data samples
        '''
        # Get training data based on status of SFA model
        if self.expanded:
            X = self.signals_exp
            num_signals = self.m_exp
        elif self.padded:
            X = self.signals_dyn
            num_signals = self.m_dyn
        else:
            X = self.signals
            num_signals = self.m

        # The offset is the average of the training signals across the samples
        X_means = X.mean(axis=1).reshape((num_signals,1))
        self.offset = X_means
        self.signals_cent = X - X_means

        return
    
    def __whiten_training(self):
        '''
        Private method used to whiten the centered training data to unit 
        variance and store the whitening matrix and data for later use
        '''
        # Scale signals to unit variance using SVD
        X = self.signals_cent
        
        # SVD of the var-cov matrix
        Sigma = np.matmul(X,X.T)
        U,Lambda,UT = np.linalg.svd(Sigma)
        
        # Calculate the whitening matrix
        Q = np.matmul(np.diag(Lambda**-(1/2)),UT)
        self.white_mat = Q
        
        Z = np.matmul(Q,X)
        self.signals_norm = Z
        
        return
    
    def __transform_training(self):
        '''
        Private method used to transform the whitened data into the slow
        features and store the transformation matrix and slow feature speeds
        '''
        
        # TODO: Replace direct eigendecompsition for SVD
        Z = self.signals_norm

        # Approximate the first order time derivative of the signals
        Zdot = Z[:,1:] - Z[:,:-1]
        
        # Eigendecompisition of the var-cov matrix of Zdot
        Zdotcov = np.matmul(Zdot,Zdot.T)
        Omega, P = np.linalg.eig(Zdotcov)

        # The loadings are ordered to find the slowest varying features
        Omega, P = self.__order(Omega,P)

        W = np.matmul(P.T,self.white_mat)
        S = np.matmul(P.T,Z)

        self.slow_features_speed = Omega
        self.trans_norm_mat = P.T
        self.trans_mat = W
        self.slow_features = S

        return

    def __order(self,eigenvalues,eigenvectors):
        '''
        Private method used for ordering a pair of related ndarrays that
        contain eigenvectors and eigenvalues in ascending order of the
        eigenvalues
        Takes two ndarrays as inputs:
        - The first is a n by 1 array containing the eigenvalues
        - The second is a n by m array containing the eigenvectors as columns
        '''
        if not isinstance(eigenvectors,np.ndarray):
            return TypeError("Expected an ndarray of eigenvectors")
        if not isinstance(eigenvalues,np.ndarray):
            raise TypeError("Expected an ndarray of eigenvalues")
        if not eigenvectors.shape[1]==eigenvalues.shape[0]:
            raise IndexError("Expected 1 eigenvalue per eigenvector: "
                              + "Received " + str(eigenvectors.shape[1])
                              + " eigenvectors and "
                              + str(eigenvalues.shape[0]) + "eigenvalues")
        # TODO: Use a more efficient sorting algorithm
        numvals = eigenvalues.shape[0]
        vals = np.empty_like(eigenvalues)
        vecs = np.empty_like(eigenvectors)
        for i in range(0,numvals):
            index = np.argmin(eigenvalues)
            vals[i] = eigenvalues[index]
            vecs[:,i] = eigenvectors[:,index]
            eigenvalues[index] = np.inf
        
        return(vals,vecs)

    def train(self,dynamic_copies=None,expansion_order=None):
        # Runs the SFA algorithm and returns the slow features
        if not dynamic_copies is None:
            if type(dynamic_copies) == int:
                if dynamic_copies > 0:
                    self.__dynamize_training(dynamic_copies)
                elif dynamic_copies == 0:
                    # No lagged copies added
                    pass
                else:
                    raise ValueError("Invalid number of lagged copies")
            else:
                raise TypeError("Expected an integer value" +
                                "for the number of lagged copies")

        if not expansion_order is None:
            if type(expansion_order) == int:
                if expansion_order > 1:
                    self.__expand_training(expansion_order)
                elif expansion_order == 1:
                    # No expansion performed
                    pass
                else:
                    raise ValueError("Invalid expansion order")
            else:
                raise TypeError("Expected an integer value " +
                                "for the order of expansion")
                
            
        self.__center_training()
        self.__whiten_training()
        self.__transform_training()
        return self.slow_features

    def partition(self,q = 0.1):
        # Partitions slow features into the slowest (1-q)*100% features
        # and the fastest q*100% features

        if self.slow_features is None:
            raise RuntimeError("Model has not been trained yet")
        # Find slowness of input signals
        xdot = self.signals[:,1:] - self.signals[:,:-1]
        # TODO: Check if this is how to find slowness of input signals
        sig_speed = np.multiply(xdot,xdot).mean(axis=1)
        sig_sorted = np.sort(sig_speed)
        
        # Find threshold value
        threshold_index = int(floor(sig_sorted.size-q*sig_sorted.size))
        threshold = sig_sorted[threshold_index]

        # Find where to partition slow features
        sf_speed = self.slow_features_speed
        for i in range(sf_speed.size):
            if sf_speed[i] > threshold:
                self.Me = i
                self.Md = sf_speed.size-i
                break

        if self.Me is None:
            # If all signals are slower than the threshold
            self.Me = 0
            self.Md = sf_speed.size

        self.slow_features_d = self.slow_features[:self.Me,:]
        self.slow_features_e = self.slow_features[self.Me:,:]
        self.parted = True
        self.calculate_crit_values()
        return

    def calculate_crit_values(self,alpha=0.01):
        # Calculate critical values for monitoring
        if alpha > 1:
            raise ValueError("Confidence level is capped at 1")

        if self.padded:
            N = self.N_dyn
        else:
            N = self.N
        p = 1-alpha
        Md = self.Md
        Me = self.Me
        
        gd = (Md*(N**2-2*N))/((N-1)*(N-Md-1))
        ge = (Me*(N**2-2*N))/((N-1)*(N-Me-1))
    
        T_d_crit = stats.chi2.ppf(p,Md)
        T_e_crit = stats.chi2.ppf(p,Me)
        S_d_crit = gd*stats.f.ppf(p,Md,N-Md-1)
        S_e_crit = ge*stats.f.ppf(p,Me,N-Me-1)

        self.T_d_crit = T_d_crit
        self.T_e_crit = T_e_crit
        self.S_d_crit = S_d_crit
        self.S_e_crit = S_e_crit
        self.gd = gd
        self.ge = ge
        
        return(T_d_crit,T_e_crit,S_d_crit,S_e_crit)

    def __dynamize_online(self, signals, num_samples):
        # Add time-delayed copies
        # copies does not include the original set of signals
        copies = self.padded_copies
        N_dyn = num_samples - copies
        X = np.zeros((self.m_dyn,N_dyn))
        for i in range(0,copies+1):
            start_row = i * self.m
            stop_row  = (i + 1) *self. m
            start_col = i
            stop_col = N_dyn + i
            X[start_row:stop_row,:] = signals[:,start_col:stop_col]
        return(X)

    def __expand_online(self,signals,num_samples):
        if self.padded:
            num_signals = self.m_dyn
        else:
            num_signals = self.m

        k = self.m_exp
        X_exp = np.empty((k,num_samples))

        # Add expanded signals
        # Order 1
        X_exp[0:num_signals,:] = X
        
        pos = num_signals # Where to add new signal
        for order in range(2,n+1):
            # Order 2 -> n
            for comb in combinations(range(num_signals),order):
                exp_signal = np.ones((1,num_samples))
                for i in comb:
                    exp_signal = exp_signal*X[i,:]
                X_exp[pos,:] = exp_signal
                pos += 1
                
        return(X_exp)

    def __center_online(self, X):
        '''
        Private method that centers a sample or set of samples based 
        on the offset that was performed on the training data
        '''
        offset = self.offset
        centered = X - offset
        return(centered)

    def __transform_online(self,X):
        '''
        Private method that transforms a centered data matrix into the
        corresponding slow features
        Takes a m by n ndarray data matrix X as input
        Returns a m by n ndarray matrix of slow features 
        '''
        W = self.trans_mat
        S = np.matmul(W,X)
        return(S)
    
    def calculate_monitors(self, online_data):
        # Takes in a batch of online data and returns an array
        # that contains the statistical indices
        # Data should have enough samples to be padded as previously done
        if not self.parted:
            raise RuntimeError("Signals have not been partitioned yet")
        if not isinstance(online_data, np.ndarray):
            raise TypeError("Expected a numpy ndarray object")
            
        if online_data.ndim == 1:
            # For 1-D arrays, assume there is one signal with multiple samples
            m = 1
            N = online_data.shape[0]
        elif online_data.ndim == 2:
            m = online_data.shape[0]
            N = online_data.shape[1]
        else:
            raise TypeError("Array input is not one or two dimensional")

        
        # Calculate slow features using transformation matrix
        if self.padded:
            online_data = self.__dynamize_online(online_data,N)
        if self.expanded:
            online_data = self.__expand_online(online_data,N)

        X = self.__center_online(online_data)
        slow_features_online = self.__transform_online(X)
        
        # Split slow features into fastest and slowest based on training data
        s_d = slow_features_online[:self.Me,:]
        s_e = slow_features_online[self.Me:,:]
        Omega_d_inv = np.diag(self.slow_features_speed[:self.Me]**(-1))
        Omega_e_inv = np.diag(self.slow_features_speed[self.Me:]**(-1))
        # Calculate time derivatives
        sdot_d = s_d[:,1:] - s_d[:,:-1]
        sdot_e = s_e[:,1:] - s_e[:,:-1]
        
        T_squared_d = np.matmul(s_d.T,s_d)
        T_squared_e = np.matmul(s_e.T,s_e)
        S_squared_d = np.matmul(np.matmul(sdot_d.T,Omega_d_inv),sdot_d)
        S_squared_e = np.matmul(np.matmul(sdot_e.T,Omega_e_inv),sdot_e)

        return(T_squared_d,T_squared_e,S_squared_d,S_squared_e)


if __name__ == "__main__":    
    # Import TEP and set up input data
    training_sets = list(imp.importTrainingSets([0]))
    testing_sets  = list(imp.importTestSets())
    training_set_0 = training_sets[0]
    testing_set_0 = testing_sets[0]
    T = testing_set_0[1].T
    X = training_set_0[1]
    X = np.delete(X,range(22,41),0)
    T = np.delete(T,range(22,41),0)
    d = 2
    
    # Run SFA
    SlowFeature = SFA(X)
    Y = SlowFeature.train(2,2)
    SlowFeature.partition(0.5)
    T_d, T_e, S_d, S_e  = SlowFeature.calculate_monitors(T)
    print(T_d.shape,T_e.shape,S_d.shape,S_e.shape)
    # Plotting
    

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(T_d[0,:])
    plt.subplot(3,1,2)
    plt.plot(T_d[:,0])
    plt.subplot(3,1,3)
    plt.plot(np.diag(T_d))
    '''
    plt.figure("Measured Variables")
    for i in range(22):
        plt.subplot(6,4,i+1)
        plt.plot(X[i,:])
    plt.figure("Manipulated Variables")
    for i in range(11):
        plt.subplot(3,4,i+1)
        plt.plot(X[22+i,:])
    
    plt.figure("9 Slowest Features")
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(Y[i,:])

    plt.figure("9 Fastest Features")
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(Y[-(i+1),:])
    ''' 
    plt.show()
