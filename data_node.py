""" Classes for processing data

Data Nodes are used to process data (nonlinear expansion, dynamization, etc.).

Classes
-------
Node: class
    Generic node for batch processing
IncrementalNode: class
    Node used for incremental processing
"""

__author__ = "Hussein Saafan"

import warnings
from itertools import combinations_with_replacement
from math import factorial

import numpy as np


class Node:
    """ Batch data processing

    Attributes
    ----------
    _num_variables: int
        The number of variables per datapoint before processing

    """
    input_variables = None
    dynamic_copies = None
    expansion_order = None
    output_variables = None

    def __init__(self, initial_num_variables, dynamic_copies, expansion_order):
        """ Class constructor

        Parameters
        ----------
        data: numpy.ndarray
            The first set of data to store in the node for processing
            must be d by n where d is the number of variables and n is
            the number of samples
        """
        self._store_input_variables(initial_num_variables)
        self._store_dynamic_copies(dynamic_copies)
        self._store_expansion_order(expansion_order)
        self._store_output_variables()

    def _store_input_variables(self, input_variables):
        """ Stores the initial number of variables before processing

        Parameters
        ----------
        input_variables: int
            Number of input variables before processing
        """
        if not isinstance(input_variables, int):
            raise TypeError(f"{type(m)} is not a valid input for input "
                            "variabes, expected int")
        elif input_variables < 1:
            raise ValueError("Expected a positive integer input for "
                             "number of variables")
        else:
            self.input_variables = input_variables
        return

    def _store_dynamic_copies(self, dynamic_copies):
        """ Stores number of dynamic copies to be used

        Parameters
        ----------
        dynamic_copies: int
            Number of lagged copies to append to data
        """
        if dynamic_copies is None:
            dynamic_copies = 0

        if isinstance(dynamic_copies, int):
            if dynamic_copies > 0:
                self.dynamic_copies = dynamic_copies
            elif dynamic_copies < 0:
                raise ValueError("Number of lagged copies must be "
                                 "positive")
            else:
                warnings.warn("No dynamization performed", RuntimeWarning)
                self.dynamic_copies = 0
        else:
            raise TypeError("Number of lagged copies must be an integer")
        return

    def _store_expansion_order(self, expansion_order):
        """ Stores the order of nonlinear expansion to be used

        Parameters
        ----------
        n: int
            Order of nonlinear expansion
        """
        if expansion_order is None:
            expansion_order = 1

        if isinstance(expansion_order, int):
            if expansion_order > 1:
                self.expansion_order = expansion_order
            elif expansion_order < 1:
                raise ValueError("Invalid expansion order")
            else:
                warnings.warn("No expansion performed", RuntimeWarning)
                self.expansion_order = 1
        else:
            raise TypeError("Expansion order must be an integer")
        return

    def _store_output_variables(self):
        """ Calculates and stores the ouput dimension """
        m = self.input_variables
        d = self.dynamic_copies
        n = self.expansion_order
        # From dynamization
        num_dyn_vars = m*(d + 1)
        # From expansion
        # Order 1
        output_vars = num_dyn_vars
        for r in range(2, n+1):
            # Order 2 -> expansion_order
            output_vars += (factorial(r + num_dyn_vars - 1)
                            / (factorial(r) * (factorial(num_dyn_vars - 1))))
        self.output_variables = int(output_vars)
        return

    def _check_input_data(self, data):
        """ Check to make sure data input is valid

        Parameters
        ----------
        data: numpy.ndarray
            The set of data to check
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Expected a numpy ndarray object")

        if data.ndim != 2:
            raise RuntimeError("Array input should be two dimensional, "
                               f"current dimensions are: {data.shape}")
        else:
            data_vars = data.shape[0]
            data_samples = data.shape[1]

        if self.input_variables is not None:
            if data_vars != self.input_variables:
                raise RuntimeError("Variables do not match existing data")

        if data_vars > data_samples and data_samples != 1:
            warnings.warn("There are more variables than samples: "
                          "Check that the data has been entered "
                          "correctly with column samples and variable rows",
                          RuntimeWarning)
        return

    def _dynamize(self, data):
        """ Appends lagged copies to data

        Creates lagged copies of the original data, where each copy is lagged
        by 1 sample more than the previous copy. The number of samples is then
        reduced by the number of copies by deleting samples from the beginning

        Parameters
        ----------
        data: numpy.ndarray
            The set of data to dynamize

        Returns
        -------
        data_dyn: numpy.ndarray
            The new set of data with the lagged copies added
        """
        d = self.dynamic_copies
        if data.shape[1] < d:
            raise RuntimeError("Not enough samples have been passed")
        elif d == 0:
            data_dyn = data
        else:
            data_dyn = np.copy(data)
            for i in range(1, d+1):
                rolled = np.roll(data, i, axis=1)
                data_dyn = np.append(data_dyn, rolled, axis=0)
            data_dyn = np.delete(data_dyn, range(d), axis=1)

        return(data_dyn)

    def _nonlinear_expansion(self, data):
        """ Performs nonlinear expansion on data

        For a set of signals [x1, x2, ... xn], performs a nonlinear expansion
        and returns [x1, x2, ..., xn, x1*x1, x1*x2, ... xn*xn, ... x1*x2*...
        *xd, ... xn^d] where d is the expansion order.

        Parameters
        ----------
        data: numpy.ndarray
            The set of data to expand

        Returns
        -------
        data_exp: numpy.ndarray
            The new set of expanded data
        """
        expansion_order = self.expansion_order

        if expansion_order == 1:
            data_exp = data
        else:
            m = data.shape[0]
            n = data.shape[1]

            data_exp = np.empty((self.output_variables, n))

            # Add expanded signals
            # Order 1
            data_exp[0:m, :] = data

            pos = m  # Where to add new signal
            for order in range(2, expansion_order + 1):
                # Order 2 -> expansion_order
                for comb in combinations_with_replacement(range(m), order):
                    exp_signal = np.ones((1, n))
                    for i in comb:
                        exp_signal = exp_signal*data[i, :]
                    data_exp[pos, :] = exp_signal
                    pos += 1

        return(data_exp)

    def process_data(self, raw_data):
        """ Processes raw data

        Dynamize and then perform nonlinear expansion on some data. The
        number of output variables will be increased if dynamizing or
        expanding. The number of output samples will be reduced if dynamizing.

        Parameters
        ----------
        raw_data: numpy.ndarray
            The set of data to process

        Returns
        -------
        processed_data: numpy.ndarray
            The new set of processed data
        """
        self._check_input_data(raw_data)
        dyn_data = self._dynamize(raw_data)
        processed_data = self._nonlinear_expansion(dyn_data)
        return(processed_data)


class IncrementalNode(Node):
    """ Incremental data processing

    Extends Node for incremental data. This class processes data one sample
    at a time. Data input must be d by 1 where d is the number of variables.

    Attributes
    ----------
    prev_samples: numpy.ndarray
        Last few samples needed to dynamize new data
    """
    prev_samples = np.empty((0, 1))

    def _dynamize(self, sample):
        """ Appends lagged copies to sample

        Append prev_samples to sample and update prev_samples. If prev_sample
        hasn't stored enough samples yet to fully dynamize, returns a signal
        with filled with zeros.

        Parameters
        ----------
        sample: numpy.ndarray
            A singal sample that is dynamized

        Returns
        -------
        sample_dyn: numpy.ndarray
            The new sample with the lagged samples added
        """
        d = self.dynamic_copies
        m = self.input_variables

        if d == 0:
            sample_dyn = sample
        else:
            sample_dyn = np.append(sample, self.prev_samples, axis=0)
            # Store at most d lagged copies in prev_samples
            self.prev_samples = sample_dyn[:d * m]

            if sample_dyn.shape[0] != m * (d + 1):
                # Not enough signals stored to fully dynamize
                sample_dyn = np.zeros(m * (d + 1))
        return(sample_dyn)

    def _nonlinear_expansion(self, sample):
        """ Performs nonlinear expansion on sample

        For a set of variables [x1, x2, ... xn], performs a nonlinear expansion
        and returns [x1, x2, ..., xn, x1*x1, x1*x2, ... xn*xn, ... x1*x2*...
        *xd, ... xn^d] where d is the expansion order.

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to expand

        Returns
        -------
        sample_exp: numpy.ndarray
            The new expanded sample
        """
        expansion_order = self.expansion_order

        if expansion_order == 1:
            sample_exp = sample
        else:
            sample_exp = np.empty((self.output_variables, 1))
            # Add expanded signals
            # Order 1
            m = sample.shape[0]
            sample_exp[0:m] = sample

            pos = m  # Where to add new signal
            for order in range(2, expansion_order + 1):
                # Order 2 -> expansion_order
                for comb in combinations_with_replacement(range(m), order):
                    variable = 1
                    for i in comb:
                        variable = variable*sample[i]
                    sample_exp[pos] = variable
                    pos += 1

        return(sample_exp)

    def process_sample(self, raw_sample):
        """ Processes raw sample

        Dynamize and then perform nonlinear expansion on a sample. The
        number of output variables will be increased if dynamizing or
        expanding.

        Parameters
        ----------
        raw_sample: numpy.ndarray
            The sample to process of shape (input_variables, 1)

        Returns
        -------
        processed_sample: numpy.ndarray
            The new processed sample
        """
        self._check_input_data(raw_sample)
        sample_dyn = self._dynamize(raw_sample)
        processed_sample = self._nonlinear_expansion(sample_dyn)

        return(processed_sample)
