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
        self.num_variables = data.shape[0]
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
                               "current dimensions are: " + str(data.shape))
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
            data_exp[0:m,:] = data
            
            pos = m # Where to add new signal
            for order in range(2,expansion_order+1):
                # Order 2 -> expansion_order
                for comb in combinations(range(m),order):
                    exp_signal = np.ones((1,n))
                    for i in comb:
                        exp_signal = exp_signal*data[i,:]
                    data_exp[pos,:] = exp_signal
                    pos += 1

        return(data_exp)

class IncrementalNode(Node):

    initial_num_variables = None
    
    dynamic_copies = None         # Number of lagged copies
    dynamic = False               # Whether any lagged copies are used or not
    prev_signals = np.empty((0,1)) # Previous signals used for dynamizing

    expansion_order = None        # Order of nonlinear expansion
    processed_num_variables = None
    
    def __init__(self, initial_num_variables, dynamic_copies, expansion_order):
        '''
        Stores dimensions of data being run through node
        initial_num_variables -> Number of variables before being dynamized and expanded
        dynamic_copies        -> Number of lagged copies to append to data
        expansion_order       -> Order of nonlinear expansion
        '''
        self._store_initial_num_variables(initial_num_variables)
        self._store_dynamic_copies(dynamic_copies)
        self._store_expansion_order(expansion_order)
        self._store_num_vars(initial_num_variables, dynamic_copies, expansion_order)
        return

    def _store_initial_num_variables(self, n):
        '''
        Checks and stores the initial data dimension
        n -> Number of variables before being dynamized and expanded
        '''
        if not type(n) == int:
            raise TypeError(str(type(n)) + " is not a valid input for number of variables," +
                            " expected int type")
        elif n < 1:
            raise ValueError("Expected a positive integer input for number of variables")
        else:
            self.initial_num_variables = n
        return
 
    def _store_dynamic_copies(self,d):
        '''
        Perform checks on number of dynamic copies entered and store value
        d -> Number of lagged copies to append to data
        '''
        if not (d is None):
            if type(d) == int:
                if d >= 0:
                    self.dynamic = True
                    self.dynamic_copies = d
                elif d == 0:
                    self.dynamic = False
                    self.dynamic_copies = d
                else:
                    raise ValueError("Number of lagged copies must be positive")
            else:
                raise TypeError("Number of lagged copies must be an integer")
        else:
            warnings.warn("Number of lagged copies not specified, using 0",
                          RuntimeWarning)
            self.dynamic = False
            self.dynamic_copies = 0
        return

    def _store_expansion_order(self,n):
        '''
        Perform checks on expansion order and store value
        n -> Order of nonlinear expansion
        '''
        if not (n is None):
            if type(n) == int:
                if n >= 1:
                    self.expansion_order = n
                else:
                    raise ValueError("Expansion order must be at least 1")
            else:
                raise TypeError("Expansion order must be an integer")
        else:
            warnings.warn("Expansion order not specified, using 1",
                          RuntimeWarning)
            self.expansion_order = 1
        return

    def _store_num_vars(self, initial_num_variables, dynamic_copies, expansion_order):
        '''
        Calculates the number of variables being input based on number of lagged copies
        and the expansion order
        initial_num_variables -> Number of variables before being dynamized and expanded
        dynamic_copies        -> Number of lagged copies to append to data
        expansion_order       -> Order of nonlinear expansion
        '''
        # From dynamization
        num_dyn_vars = initial_num_variables*(dynamic_copies + 1) 
        # From expansion
        # Order 1
        num_vars = num_dyn_vars
        for r in range(2,expansion_order+1):
            # Order 2 -> expansion_order
            num_vars += fac(r + num_dyn_vars - 1)/(fac(r) * (fac(num_dyn_vars - 1)))
        self.processed_num_variables = int(num_vars)
        return

    def _check_input_data(self,raw_signal):
        '''
        Performs a variety of checks on the input data to make sure that its valid
        raw_signal -> Signal before any processing
        '''
        if not isinstance(raw_signal,np.ndarray):
            raise TypeError("Expected a numpy ndarray object")
        if not raw_signal.shape == (self.initial_num_variables, 1):
            raise RuntimeError("Expected input of shape (" + str(self.initial_num_variables) + 
                                ",1) " + "current dimensions are: " + str(raw_signal.shape))
        return
 
    def _dynamize(self,raw_signal):
        '''
        Adds lagged copies to signal
        raw_signal -> One raw data point before any processing
        dyn_signal -> Output signal after dynamizing
        is_full    -> Have enough data to get signal with lagged copies 
        '''
        if self.dynamic:
            lagged_copies_length = self.dynamic_copies*self.initial_num_variables
            dyn_signal = np.append(raw_signal,self.prev_signals, axis=0)
            self.prev_signals = dyn_signal[:lagged_copies_length]
            if dyn_signal.shape[0] == lagged_copies_length + self.initial_num_variables:
                is_full = True
            else:
                is_full = False
        else:
            dyn_signal = raw_signal
            is_full = True
 
        return(dyn_signal, is_full)
    
    def _nonlinear_expansion(self, dyn_signal):
        '''
        Performs nonlinear expansion on signal
        '''
        if self.expansion_order == 1:
            exp_signal = dyn_signal
        else:
            exp_signal = np.empty((self.processed_num_variables,1))
            # Add expanded signals
            # Order 1
            m = dyn_signal.shape[0]
            exp_signal[0:m] = dyn_signal
            
            pos = m # Where to add new signal
            for order in range(2,self.expansion_order+1):
                # Order 2 -> expansion_order
                for comb in combinations(range(m),order):
                    variable = 1
                    for i in comb:
                        variable = variable*dyn_signal[i]
                    exp_signal[pos] = variable
                    pos += 1

        return(exp_signal)
    
    def process_signal(self, raw_signal):
        '''
        Dynamizes and expands input signal
        raw_signal       -> One raw data point before any processing
        processed_signal -> Output signal after dynamizing and expanding
        is_full          -> Have enough data to get signal with lagged copies
        '''
        self._check_input_data(raw_signal)
        dyn_signal, is_full = self._dynamize(raw_signal)
        if is_full:
            processed_signal = self._nonlinear_expansion(dyn_signal)
        else:
            processed_signal = dyn_signal
        
        return(processed_signal, is_full)