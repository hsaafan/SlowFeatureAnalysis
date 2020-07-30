""" Recursive Slow Feature Analysis

Implementation of Recursive Slow Feature Analysis [1]_

Classes
-------
RSFA: class
    The class used for Recursive Slow Feature Analysis

References
----------
.. [1] Shang, C., Yang, F., Huang, B., & Huang, D. (2018). Recursive Slow
       Feature Analysis for Adaptive Monitoring of Industrial Processes. IEEE
       Transactions on Industrial Electronics (1982), 65(11), 8895–8905.
       <https://doi.org/10.1109/tie.2018.2811358>
"""
__author__ = "Hussein Saafan"

import numpy as np
import numpy.linalg as LA
import scipy.stats as stats
import matplotlib.pyplot as plt

import tep_import as imp
from data_node import IncrementalNode
from standardization_node import RecursiveStandardization


class RSFA(IncrementalNode):
    """ Recursive Slow Feature Analysis

    Attributes
    ----------
    time: int
        The current time index which increments by 1 after each iteration
    delta: float
        The time between samples in the data, default is 1
    covariance_delta: numpy.ndarray
        The estimated covariance matrix of the derivative of the samples
    num_features: int
        The amount of features to calculate.
    standardization_node: RecursiveStandardization
        Object used to standardize input samples
    centered_previous: numpy.ndarray
        The previous sample that has been centered
    centered_current: numpy.ndarray
        The current sample that has been centered
    centered_delta_current: numpy.ndarray
        The current centered sample derivative
    centered_delta_previous: numpy.ndarray
        The previous centered sample derivative
    feature_previous: numpy.ndarray
        The previous feature output
    feature_current: numpy.ndarray
        The current feature output
    transformation_matrix: numpy.ndarray
        The matrix that transforms whitened data into the features.
    """
    time = 0
    delta = 1
    covariance_delta = None
    num_features = None
    transformation_matrix = None
    standardization_node = None
    centered_previous = None
    centered_current = None
    centered_delta_current = None
    centered_delta_previous = None
    feature_previous = None
    feature_current = None

    def __init__(self, input_variables, num_features,
                 expansion_order=None, dynamic_copies=None):
        """ Class constructor

        Parameters
        ----------
        input_variables: int
            Number of variables to be input before any processing
        num_features: int
            Desired number of slow features to calculate
            FIXME: currently only works if num_features is equal to number of
            variables after processing. Has to do with P and Q dimensions.
        expansion_order: int
            The order of nonlinear expansion to perform
        dynamic_copies: int
            The number of lagged copies to add to the data
        """
        super().__init__(input_variables, dynamic_copies, expansion_order)
        K = self.output_variables  # Number of variables after proccessing
        J = num_features

        self.num_features = J
        self.covariance_delta = np.random.randn(K, K)
        self.transformation_matrix = np.random.randn(K, J)
        self.centered_delta_current = np.zeros((K, 1))
        self.centered_delta_previous = np.zeros((K, 1))
        self.speeds = np.random.random_sample(J)

    def _update_delta_cov(self, x_dot, x_dot_prev, eta):
        """ Updates the covariance matrix for the derivative

        Parameters
        ----------
        x_dot: numpy.ndarray
            The current centered sample derivative
        x_dot_prev: numpy.ndarray
            The previous centered sample derivative
        eta: float
            The learning rate
        """
        history = (1 - eta) * self.covariance_delta
        update = eta * (x_dot @ x_dot.T)
        self.covariance_delta = history + update

        return(self.covariance_delta)

    def _update_transformation_matrix(self, x_dot, x_dot_prev, eta, Q):
        """ Updates the estimate of the transformation matrix

        Parameters
        ----------
        x_dot: numpy.ndarray
            The current centered sample derivative
        x_dot_prev: numpy.ndarray
            The previous centered sample derivative
        eta: float
            The learning rate
        Q: numpy.ndarray
            The current estimate of the whitening matrix
        """
        cov_delta = self._update_delta_cov(x_dot, x_dot_prev, eta)
        gam = 10
        P = self.transformation_matrix
        S = np.eye(self.output_variables) - (Q @ cov_delta @ Q.T) / gam
        P, R = LA.qr(S @ P)
        self.transformation_matrix = P
        W = P @ Q
        '''
        # Transformtion using SVD
        # Working, don't change...
        PT, L, P = LA.svd(Q @ cov_delta @ Q.T, hermitian=True)
        self.speeds = (1 - eta) * self.speeds + eta * L
        self.transformation_matrix = P
        W = P @ Q
        '''
        return(W)

    def update_control_limits(self, sample, sample_delta, alpha):
        # FIXME: Depreceated, need to implement new version
        # Calculate limits
        x = sample
        x_dot = sample_delta
        c = stats.norm.cdf(alpha)
        t1 = self.th_1
        t2 = self.th_2
        t3 = self.th_3
        h = 1 - (2*t1*t3)/(3*(t2**2))
        Q = t1*(1 + (c*(2*t2*(h**2))**(1/2)/t1) + t2*h*(h-1)/(t1**2)) ** (1/h)

        # Calculate test stats
        ident = np.eye(self.Pd.shape[0], self.Pd.shape[1])
        T_sqr = x.T @ self.Q.T @ self.Pd.T @ self.Pd @ self.Q @ x
        T_e_sqr = x.T @ self.Q.T @ (ident - self.Pd.T @ self.Pd) @ self.Q @ x
        S_sqr = x_dot.T @ self.Q.T @ self.Pd.T @ self.Pd @ self.Q @ x_dot

        stats = [T_sqr, T_e_sqr, S_sqr]
        return(stats)

    def add_data(self, raw_sample):
        """Update the RSFA model with new data points

        Parameters
        ----------
        raw_sample: numpy.ndarray
            The sample used to update the model and calculate the features

        Returns
        -------
        y: numpy.ndarray
            The calculated features for the input sample
        stats: array
            An array containing the monitoring statistics in the following
            order: T^2 (for slowest features), T^2 (for fastest features),
            S^2 (for slowest features).
        """
        # Default values
        y = np.zeros((self.num_features, 1))
        stats = [0, 0, 0]

        """ Signal preprocessing """
        raw_sample = raw_sample.reshape((raw_sample.shape[0], 1))
        sample = self.process_sample(raw_sample)
        if np.allclose(sample, 0):
            # Need more data to get right number of dynamic copies
            return(y, stats)

        """ Update Learning rate """
        # TODO: Set learning rate schedule
        eta = np.max([1 / (self.time + 2), 1e-4])

        """ Signal centering and whitening """
        if self.time > 0:
            # Updates centered signals stored
            self.centered_previous = np.copy(self.centered_current)
            x_prev = self.centered_previous
        else:
            # On first pass through, create the standardization object
            node = RecursiveStandardization(sample)
            self.standardization_node = node

        """ TODO: Move this section into RecursiveNode class """
        x = self.standardization_node._center(sample, eta)
        self.standardization_node._update_sample_cov(x, eta)
        self.standardization_node._whiten(x, eta)
        Q = self.standardization_node.whitening_matrix
        """ ------------------------------------------------ """

        self.centered_current = x
        self.z = Q @ x  # TODO: Remove this after testing is done
        """ Transformation """
        if self.time > 0:
            # Update derivatives
            x_dot_prev = np.copy(self.centered_delta_previous)
            x_dot = (x - x_prev)/self.delta
            self.centered_delta_current = x_dot
            self.centered_delta_previous = x_dot_prev

            # Transform data
            W = self._update_transformation_matrix(x_dot, x_dot_prev, eta, Q)
            y = W @ x

            # Update stored features
            self.feature_previous = np.copy(self.feature_current)
            self.speeds = (1 - eta) * self.speeds + np.diag(y @ y.T)
            y_prev = self.feature_previous
            self.feature_current = y

        """ Update monitoring stats """
        # TODO: Add this

        """ Update model and return """
        self.time += 1

        return(y, stats)
