import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement as combinations
from math import factorial as fac

class SFA:
    '''
    raw_signals is an m-dimensional input signal:
       x(t) = [x1(t), x2(t), ..., xm(t)]^T
    m is the number of different signals
    N is the number of data samples available in each signal
    normalized_signals is raw_signals normalized with 0 mean and unit variance
    '''
    
    
    raw_signals = None
    m = None
    N = None
    
    normalized_data = None
    raw_expanded_data = None
    normalized_expanded_data = None
    slow_features = None

    whitening_matrix = None
    expansion_order = None
    def __init__(self, raw_signals, expansion_order=2):
        if not isinstance(raw_signals, np.ndarray):
            raise TypeError("Expected a numpy ndarray object")

        if raw_signals.ndim == 1:
            self.N = raw_signals.shape[0]
            self.m = 1
        else:
            self.N = raw_signals.shape[1]
            self.m = raw_signals.shape[0]

        self.raw_signals = raw_signals
        self.expansion_order = expansion_order
        return
    
        
    def normalize(self):
        # Normalize input signals to 0 mean and unit variance
        X = self.raw_signals
        # Mean and stdv of each signal
        X_means = X.mean(axis=1).reshape((self.m,1))
        X_stdv  = X.std(axis=1).reshape((self.m,1))

        self.normalized_data = (X-X_means)/X_stdv

        return
    
    def expansion(self):
        # Perform nth order nonlinear expansion of signal
        n = self.expansion_order
        if not type(n) == int:
            raise TypeError("Expected an integer for the expansion order")
        if not n > 1:
            raise ValueError("Can't expand with order less than 2")

        k = self.m # Order 1
        for r in range(2,n+1):
            # Order 2 -> n
            k += fac(r+self.m-1)/(fac(r)*(fac(self.m-1)))
        k = int(k)
            
        Ztilda = np.empty((k,self.N))
        X = self.normalized_data
        
        # Order 1
        Ztilda[0:self.m,:] = X
        
        counter = self.m
        for order in range(2,n+1):
            # Order 2 -> n
            for comb in combinations(range(self.m),order):
                exp_signal = np.ones((1,self.N))
                for i in comb:
                    exp_signal = exp_signal*X[i,:]
                Ztilda[counter,:] = exp_signal
                counter += 1
        
        # Set the mean to 0
        Ztilda = Ztilda - Ztilda.mean(axis=1).reshape(k,1)
        self.raw_expanded_data = Ztilda
        return


    def whitening(self):
        # Expanded signal is whitened
        Ztilda = self.raw_expanded_data
        
        # SVD of the var-cov matrix
        Sigma = np.matmul(Ztilda,Ztilda.T)
        U,Lambda,UT = np.linalg.svd(Sigma)

        # Calculate the whitening matrix
        Q = np.matmul(np.diag(Lambda**-(1/2)),UT)
        self.whitening_matrix = Q
        
        self.normalized_expanded_data = np.matmul(Q,Ztilda)
        return
    
    def compute_slow_features(self):
        # Step 5: Principal component analysis
        # TODO: Replace direct eigendecompsition for a more efficient
        # numerical approach
        Z = self.normalized_expanded_data
        
        # Approximate the first order time derivative of the signals
        Zdot = Z[:,1:] - Z[:,:-1]
        
        # Eigendecompisition of the var-cov matrix of Zdot
        Zdotcov = np.matmul(Zdot,Zdot.T)
        Omega, P = np.linalg.eig(Zdotcov)

        # The loadings are ordered to find the slowest varying features
        P, Omega = self.order(P,Omega)
        
        self.slow_features = np.matmul(P.T,Z)
        return

    def order(self,eigenvectors,eigenvalues):
        # Sort and return the eigenvectors in ascending order according
        # to their eigenvalues
        if not isinstance(eigenvectors,np.ndarray):
            return TypeError("Expected an ndarray of eigenvectors")
        if not isinstance(eigenvalues,np.ndarray):
            return TypeError("Expected an ndarray of eigenvalues")
        if not eigenvectors.shape[1]==eigenvalues.shape[0]:
            return IndexError("Expected 1 eigenvalue per eigenvector: "
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
        
        return(vecs,vals)
    
    def train(self,expand=True):
        # Runs the SFA algorithm and returns the slow features
        # TODO: Once more options are available, expand this function
        #       to use those options (e.g. cubic expansion)
        self.normalize()
        if expand:
            self.expansion()
        else:
            self.raw_expanded_data = self.normalized_data
        self.whitening()
        self.compute_slow_features()
        
        return self.slow_features


if __name__ == "__main__":    
    ###########################################
    # Setting up a data sample
    # Example taken from:
    # https://towardsdatascience.com/a-brief-introduction-to-slow-feature-analysis-18c901bc2a58
    
    length = 300
    S = np.zeros((length,1),'d')
    D = np.zeros((length,1),'d')
    S[0] = 0.6
    for t in range(1,length):
        D[t] = np.sin(np.pi/75. * t) - t/150.
        S[t] = (3.7+0.35*D[t]) * S[t-1] * (1 - S[t-1])

    
    k = 10 # Number of time delayed copies to add
    X = np.zeros((k,length-(k-1)),'d')
    for i in range(0,k):
       X[i,:] = S[i:length+i+1-k,0]
    ###########################################
    
    # Run SFA
    SlowFeature = SFA(X)
    Y = SlowFeature.train().T

    
    # Plotting
    plt.figure()

    plt.subplot(2,2,1)
    plt.title("Signal S(D) (Input)")
    plt.plot(S)
    
    plt.subplot(2,2,2)
    plt.title("Signal D")
    plt.plot(D)

    D_norm = (D-D.mean())/(D.std())
    plt.subplot(2,2,3)
    plt.title("Normalized D vs Slowest Feature")
    plt.scatter(Y[:,0],D_norm[k-1:])

    plt.subplot(2,2,4)
    plt.title("Slowest feature")
    plt.plot(Y[:,0])
    
    plt.show()
