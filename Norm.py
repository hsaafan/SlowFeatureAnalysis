import numpy as np
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
    m = None
    n = None
    
    def __init__(self,data):
        '''
        Constructor method that stores the data and normalizes it
        Other data sample can be normalized using the same offset
        and whitening matrix calculated for this data by calling the
        normalize_similar() method
        '''
        self._check_input_data(data)
        self.data = np.copy(data)
        self.m = self.data.shape[0]
        self.n = self.data.shape[1]

        return
    
    def _center(self,data):
        '''
        Centers a data matrix around 0
        '''
        data_means = data.mean(axis=1).reshape((self.m,1))
        self.offset = data_means
        centered_data = data - self.offset
        return(centered_data)
    
    def _whiten(self,data):
        ''' 
        Scale signals to unit variance using SVD
        '''
        
        # SVD of the var-cov matrix
        Sigma = np.matmul(data,data.T)/(self.n-1)
        U,Lambda,UT = np.linalg.svd(Sigma)

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
        centered_data = np.copy(data)-self.offset
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
        centered_data = np.copy(data) - self.offset
        normalized_data = np.matmul(self.white_mat,centered_data) 
        return(normalized_data)
    
