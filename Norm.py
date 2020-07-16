import numpy as np
import scipy.linalg as LA
import scipy
import warnings
from DataNode import Node

class Norm(Node):
    '''
    Scales data to 0 mean and unit variance (identity covariance matrix)
    by calculating the offset and whitening matrix
    '''
    data = None
    offset = None
    white_mat = None
    variables = None
    samples = None
    
    def __init__(self,data):
        '''
        Constructor method that stores the data and normalizes it
        Other data sample can be normalized using the same offset
        and whitening matrix calculated for this data by calling the
        normalize_similar() method
        '''
        self._check_input_data(data)
        self.data = data
        self.variables = self.data.shape[0]
        self.samples = self.data.shape[1]

        return
    
    def _center(self,data):
        '''
        Centers a data matrix around 0
        '''
        data_means = data.mean(axis=1).reshape((self.variables,1))
        self.offset = data_means
        centered_data = data - self.offset
        return(centered_data)
    
    def _whiten(self,data):
        ''' 
        Scale signals to unit variance using SVD
        '''        
        # SVD of the var-cov matrix
        Sigma = np.cov(data)
        U,Lambda,UT = np.linalg.svd(Sigma,hermitian=True)

        # Calculate the whitening matrix
        Q = np.matmul(np.diag(Lambda**-(1/2)),UT)
        self.white_mat = Q
        whitened_data = np.matmul(Q,data)

        return(whitened_data)

    def normalize(self):
        '''
        Scales the original 
        '''
        centered_data = self._center(self.data)
        normalized_data = self._whiten(centered_data)
        return(normalized_data)
    
    def center_similar(self,data):
        centered_data = data-self.offset
        return(centered_data)

    def whiten_similar(self,data):
        Q = self.white_mat
        whitened_data = np.matmul(Q,np.copy(data))
        return(whitened_data)
    
    def normalize_similar(self, data):
        '''
        Takes in a new data sample, centers and whitens it according to
        how the original data input was normalized
        '''
        self._check_input_data(data)
        centered_data = data - self.offset
        normalized_data = np.matmul(self.white_mat,centered_data) 
        return(normalized_data)


class IncrementalNorm:
    mean = None
    var = None
    sumsqrs = None

    count = None
    num_vars = None
    V = None
    eps  = 1e-8

    K = None
    L = None
    
    def __init__(self, first_signal, K, L):
        self.mean = np.zeros_like(first_signal)
        self.var = np.ones_like(first_signal)
        self.sumsqrs = np.zeros_like(first_signal) + self.eps

        self.num_vars = first_signal.shape[0]

        self.L = L
        self.K = K
        self.count = 1

        # Initialize V
        self.V = np.random.randn(self.num_vars,self.K)

    def _amnesiac(self):
        '''
        Gets the amnesiac weights based on count and a parameter
        n -> Total datapoints used so far
        L -> Amnesiac parameter (typically 2 - 4)
        '''
        n = self.count
        L = self.L
        w1 = (n - 1 - L) / n
        w2 = (1 + L) / n
        return(w1, w2)

    def _center(self, x, eta):
        '''
        Centers a signal and updates the mean
        x   -> Input signal
        eta -> Learning rate
        u   -> Centered signal
        '''
        w1, w2 = self._amnesiac()
        
        self.mean = w1 * self.mean +  w2 * x
        u = x - self.mean

        if np.allclose(u,np.zeros_like(u),atol=self.eps):
            # Add some noise if u is all zeros to prevent div by 0 errors
            u += self.eps * np.random.randn(self.num_vars, 1)
        return(u)

    def _CCIPA(self, V, u):
        '''
        Candid Covariance Free Incremental Principal Component Analysis
        V -> Previous estimate eigenvectors and eigenvalues where
               v_i = eigenvalue_i * eigenvector_i
        K -> How many eigenvectors to calculate
        u -> Zero mean signal
        '''
        w1, w2 = self._amnesiac() # Get weight parameters
        
        max_iter = int(np.min([self.K, self.count]))

        for i in range(max_iter):
            if i == self.count - 1:
                V[:,i] = u.reshape((self.num_vars))
            else:
                # Get previous estimate of v
                prev_v = V[:,i].reshape((self.num_vars, 1))

                # Get new estimate of v
                new_v = w1 * (prev_v) + w2 * ( u * ((u.T @ prev_v) / LA.norm(prev_v)) )

                # Store new estimate
                V[:,i] = new_v.reshape((self.num_vars))
                
                # Update u for next vector
                new_v_normed = new_v / (LA.norm(new_v) + self.eps)
                u = u - (u.T @  new_v_normed ) * new_v_normed
        return(V)

    def _get_whitening_matrix(self,V):
        '''
        Calculate the PCA whitening matrix
        V   -> Eigenvectors and eigenvalues estimates where
               v_i = eigenvalue_i * eigenvector_i
        S   -> Whitening matrix
        '''

        D = np.zeros((V.shape[1])) # Inverse square root of eigenvalues 
        U = np.zeros_like(V)       # Eigenvectors

        for i in range(self.K):
            lam = LA.norm(V[:,i]) + self.eps
            U[:,i] = V[:,i] / lam
            D[i] = lam ** (-1/2)
        
        S = U @ np.diag(D)

        return(S)
    
    def normalize_online(self, signal, eta):
        '''
        Centers and whitens input signal incrementally
        signal          -> Current signal
        eta             -> Learning rate
        normalized_data -> Current signal after being transformed to approximate
                           0 mean and identity matrix variance
        '''
        centered_data = self._center(signal, eta)
        self.V = self._CCIPA(self.V, centered_data)
        S = self._get_whitening_matrix(self.V)
        normalized_data = S.T @ centered_data

        # Update data count
        self.count += 1
        return(normalized_data)

    def update_CCIPA(self, signal, eta):
        '''
        Updates and returns eigenvectors without transforming data
        signal -> Current signal
        eta    -> Learning rate
        V      -> Current eigenvectors
        '''
        centered_data = self._center(signal, eta)
        self.V = self._CCIPA(self.V, centered_data)
        # Update data count
        self.count += 1
        return(self.V)