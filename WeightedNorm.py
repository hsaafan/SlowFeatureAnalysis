import numpy as np
import scipy
import warnings
from DataNode import Node

class WeightedNorm(Node):
    '''
    Scales data to 0 mean and unit variance (identity covariance matrix)
    with weighted data
    '''
    data = None          # Data input
    variables = None     # Number of variables passed
    samples = None       # Number of samples passed
    
    blocks = None        # Variable blocks
    weights = None       # Data weights
    weights_total = None # Sum of weights
    weighted_data = None # Data multiplied by weights

    offset = None
    stdv = None
    
    def __init__(self,data,blocks):
        '''
        Constructor method that stores the data and blocks
        '''
        self._check_input_data(data)
        self.data = data
        self.variables = self.data.shape[0]
        self.samples = self.data.shape[1]
        
        self._check_var_blocks(blocks,self.variables)
        self.blocks = blocks
        return
    
    def _check_var_blocks(self, blocks, num_vars):
        '''
        Checks that all the variables belong to exactly one block
        '''
        all_vars = list(range(num_vars))
        blocked_vars = []

        # Check that variables have been passed in blocks and stores them
        if hasattr(type(blocks), '__iter__'):
            for group in blocks:
                if hasattr(type(group), '__iter__'):
                    for var in group:
                        if type(var) == int:
                            blocked_vars.append(var)
                        else:
                            raise TypeError("Variables must be integers")
                else:
                    raise TypeError("Variable block must be iterable")    
        else:
            raise TypeError("Expected an iterable object of variable blocks")

        # Check that all variables have been passed exactly once  
        if all_vars == sorted(blocked_vars):
            pass
        elif len(all_vars) == len(blocked_vars):
            raise ValueError("Duplicated or invalid variables passed")
        else:
            raise ValueError("Missing or extra variables passed")
        return

    def update(self, weights):
        '''
        Takes a new set of weights and recalculates normalized data
        '''
        self._check_weights(weights)
        self.weights = weights
        self.weights_total = np.sum(weights)

        # Center and normalize weighted data
        centered_data = self._center(self.data, weights)
        normalized_data = self._whiten(centered_data, total_weight)
        
        return(normalized_data)
    
    def _check_weights(self, weights):
        '''
        Checks that there is exactly one weight per sample
        '''
        if isinstance(weights,np.ndarray):
            if weights.ndim == 1:
                if wieghts.size == self.samples:
                    # Weights are good, do nothing
                    pass
                else:
                    raise ValueError("Each sample must have one weight")
            else:
                raise TypeError("Weights must be a 1D numpy ndarray object")
        else:
            raise TypeError("Weights must be a 1D numpy ndarray object")
        return
    
    def _center(self, data):
        '''
        Centers each variable block around 0
        '''
        offset = np.empty((self.variables,1))

        # Multiply data by weights
        weights_matrix = np.multiply(self.weights,np.ones_like((data)))
        weighted_data = np.multipy(weights,data)
        for group in self.blocks:
            grouped_data = np.empty((len(group),self.samples))

            # Create matrix of all data in group
            row = 0
            for variable in group:
                grouped_data[row,:] = weighted_data[variable,:]
                row += 1

            # Find mean of group and store it for each variable in group
            group_mean = np.sum(grouped_data)/(len(group)*self.weights_total)
            for variable in group:
                offset[variable] = group_mean

        self.offset = offset
        centered_data = data - self.offset 

        return(centered_data)

    def calculate_variance(self,centered_data_1,centered_data_2=None,weights):
        '''
        Calculate the variance for each block
        '''
        if centered_data_2 is None:
            centered_data_2 = centered_data_1

        variance = np.empty((self.variables,1))
        data = np.multiply(centered_data_1,centered_data_2)
        
        # Multiply data by weights
        weights_matrix = np.multiply(weights,np.ones_like((data)))
        weighted_data = np.multipy(weights_matrix,data)

        for group in self.blocks:
            grouped_data = np.empty((len(group),self.samples))

            # Create matrix of all data in group
            row = 0
            for variable in group:
                grouped_data[row,:] = weighted_data[variable,:]
                row += 1

            # Find variance of group and store it for each variable in group
            group_var = np.sum(grouped_data)/(len(group)*self.weights_total)
            for variable in group:
                variance[variable] = group_var

        return(variance)
    
    def _whiten(self, centered_data):
        ''' 
        Scale blocks to unit variance if they have been centered
        '''
        variance = self.calculate_variance(centered_data, None, self.weights)
        stdv = variance ** (-1/2)
        whitened_data = centered_data / stdv
        
        return(whitened_data)
