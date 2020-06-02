import numpy as np
import warnings

from itertools import combinations_with_replacement as combinations
from math import factorial as fac


class Node:
    '''
    Class that contains commonly used methods to prevent cluttering
    or other classes and code duplication
    '''
    num_variables = None
    def __init__(self,data):
        self._check_input_data(data)
        self.num_variables = np.copy(data).shape[0]
        return

    def _check_input_data(self,data):
        '''
        Performs a variety of checks on the input data to make sure that
        its valid
        '''
        if not isinstance(data,np.ndarray):
            raise TypeError("Expected a numpy ndarray object")
        if not data.ndim == 2:
            raise RuntimeError("Array input should be two dimensional, " +
                               "current dimensions are: " + data.shape)
        if not self.num_variables is None:
            if not data.shape[0] == self.num_variables:
                raise RuntimeError("Variables do not match existing data")
        if data.shape[0] > data.shape[1]:
            warnings.warn("There are more signals than samples: " +
                          "Check that the data has been entered " +
                          "correctly with column samples and variable rows",
                          RuntimeWarning)
        return

    def dynamize(self, data, copies):
        '''
        Adds lagged copies to the original training signals
        Copies are lagged by 1 sample each time
        '''
        self._check_input_data(data)
        # Checks whether to dynamize or not and whether a valid number of
        # copies has been added
        if not copies is None:
            if type(copies) == int:
                if copies > 0:
                    # Run the rest of the method to dynamize
                    finish_method = False
                elif copies == 0:
                    # No lagged copies added, finish method
                    finish_method = True
                else:
                    raise ValueError("Invalid number of lagged copies")
            else:
                raise TypeError("Expected an integer value" +
                                "for the number of lagged copies")
        else:
            # No lagged copies added, finish method
            finish_method = True

        if finish_method:
            warnings.warn("No lagged copies have been added", RuntimeWarning)
            data_dyn = data
        else:
            data_dyn = np.copy(data)

            for i in range(1,copies+1):
                rolled  = np.roll(data,i,axis=1)
                data_dyn = np.append(data_dyn,rolled,axis=0)

            data_dyn = np.delete(data_dyn,range(copies),axis=1)

        return(data_dyn)
    
    
    def nonlinear_expansion(self, data, expansion_order):
        '''
        Performs nonlinear expansion on signals where n is the
        order of expanson performed
        '''
        self._check_input_data(data)
        if not expansion_order is None:
            if type(expansion_order) == int:
                if expansion_order > 1:
                    # Run the rest of the method to expand
                    finish_method = False
                elif expansion_order == 1:
                    # No expansion performed, finish method
                    finish_method = True
                else:
                    raise ValueError("Invalid expansion order")
            else:
                raise TypeError("Expected an integer value " +
                                "for the order of expansion")
        else:
            # No expansion performed, finish method
            finish_method = True

        if finish_method:
            warnings.warn("No expansion has been performed", RuntimeWarning)
            data_exp = data
        else:
            m = data.shape[0]
            n = data.shape[1]
            
            # Find dimensions of expanded matrix
            k = m # Order 1
            for r in range(2,expansion_order+1):
                # Order 2 -> expansion_order
                k += fac(r+m-1)/(fac(r)*(fac(m-1)))
            k = int(k)

            data_exp = np.empty((k,n))


            # Add expanded signals
            # Order 1
            data_exp[0:m,:] = X

            pos = n # Where to add new signal
            for order in range(2,expansion_order+1):
                # Order 2 -> expansion_order
                for comb in combinations(range(m),order):
                    exp_signal = np.ones((1,m))
                    for i in comb:
                        exp_signal = exp_signal*X[i,:]
                    data_exp[pos,:] = exp_signal
                pos += 1

        return(data_exp)

