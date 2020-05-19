import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt


class SFA:

    raw_data = None
    n = None
    d = None
    normalized_data = None
    raw_expanded_data = None
    normalized_expanded_data = None
    slow_features = None

    def __init__(self, raw_data):
        '''
        Takes an n by d numpy ndarray as input where d is the number of 
        variables and n is the number of samples
        '''
        if not isinstance(raw_data, np.ndarray):
            raise TypeError("Expected a numpy ndarray object")

        # Stores dimensions and data
        # Step 1: Get input signals
        self.n = raw_data.shape[0]
        if raw_data.ndim == 1:
            self.d = 1
        else:
            self.d = raw_data.shape[1]

        self.raw_data = raw_data
        return
    
        
    def normalize(self):
        # Step 2: Normalize input signals
        signal_means = self.raw_data.mean(axis=0)
        signal_stdv  = self.raw_data.std(axis=0)

        self.normalized_data = (self.raw_data - signal_means)/signal_stdv
        return

    # Step 3: Perform nonlinear expansion
    # TODO: Add other expansion types or generalize to any order of expansion
    def expansion_2(self):
        # Quadratic expansion
        k = int(self.d+self.d*(self.d+1)/2) # Number of expanded signals
        counter = 0
        
        self.raw_expanded_data = np.empty((self.n,k))

        # Order 1: Add all normalized signals
        for i in range(0,self.d):
            self.raw_expanded_data[:,counter] = self.normalized_data[:,i]
            counter += 1

        # Order 2: Add all products of 2 normalized signals
        for i in range(0,self.d):
            for j in range(i,self.d):
                # Start loop at i to prevent duplicates
                self.raw_expanded_data[:,counter] = ( self.normalized_data[:,i]
                                                    * self.normalized_data[:,j])
                counter += 1
        return


    def whitening(self):
        # Step 4: Whitening
        # Data is whitened using SVD of var-cov matrix

        # Set the mean = 0
        Ztilda = self.raw_expanded_data
        Ztilda = Ztilda-Ztilda.mean(axis=0)

        # Var-cov matrix
        Sigma = np.matmul(Ztilda.T,Ztilda)

        E,D,ET = np.linalg.svd(Sigma)

        # Calculate the whitening matrix
        W = np.matmul(np.diag(D**-(1/2)),ET)
        
        self.normalized_expanded_data = np.matmul(W,Ztilda.T)
        return
    
    def compute_slow_features(self):
        # Step 5: Principal component analysis
        # TODO: Replace direct eigendecompsition for a more efficient
        # numerical approach
        
        Z = self.normalized_expanded_data

        # Approximate the first order time derivative of the signals
        Zdot = Z[:,1:] - Z[:,:-1]

        
        Zdotcov = np.matmul(Zdot,Zdot.T)
        Lambda, W = np.linalg.eig(Zdotcov)

        # The loadings are ordered to find the slowest varying features
        W,Lambda = self.order(W,Lambda)
        
        self.slow_features = np.matmul(W.T,Z)
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
    
    def train(self):
        # Runs the SFA algorithm and returns the slow features
        # TODO: Once more options are available, expand this function
        #       to use those options (e.g. cubic expansion)
        self.normalize()
        self.expansion_2()
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

    
    k = 5 # Number of time delayed copies to add
    X = np.zeros((length-(k-1),k),'d')
    for i in range(0,k):
       X[:,i] = S[i:length+i+1-k,0]
    ###########################################
    
    # Run SFA
    
    SlowFeature = SFA(X)
    Y = SlowFeature.train()


    # Plotting
    plt.figure()

    plt.subplot(2,2,1)
    plt.title("Signal S(D) (Input)")
    plt.plot(S)
    
    plt.subplot(2,2,2)
    plt.title("Signal D")
    plt.plot(D)
    
    plt.subplot(2,2,3)
    D_norm = (D-D.mean())/(D.std())
    plt.title("Normalized D vs Slowest Feature")
    plt.scatter(Y.T[:,0],D_norm[k-1:])

    plt.subplot(2,2,4)
    plt.title("Slowest feature")
    plt.plot(Y.T[:,0])
    
    plt.show()
