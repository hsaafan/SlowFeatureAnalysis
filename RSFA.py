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
import scipy.stats as ST
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
    evaluation: boolean
        Whether the model is being updated or only evaluated
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
    consecutive_faults: int
        The current number of faults that have been detected
    required_faults: int
        The number of faults required for the model to update
    """
    time = 0
    delta = 1
    evaluation = False
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
    consecutive_faults = 0
    required_faults = 10

    def __init__(self, input_variables, num_features,
                 expansion_order=None, dynamic_copies=None):
        """ Class constructor

        Parameters
        ----------
        input_variables: int
            Number of variables to be input before any processing
        num_features: int
            Desired number of slow features to calculate
        expansion_order: int
            The order of nonlinear expansion to perform
        dynamic_copies: int
            The number of lagged copies to add to the data
        """
        super().__init__(input_variables, dynamic_copies, expansion_order)
        K = self.output_variables  # Number of variables after proccessing
        J = num_features

        self.Md = 55  # TODO: Replace this
        self.Me = J - self.Md
        self.num_features = J
        self.covariance_delta = np.random.randn(K, K)
        self.transformation_matrix = np.random.randn(K, J)
        self.centered_delta_current = np.zeros((K, 1))
        self.centered_delta_previous = np.zeros((K, 1))

    def _update_delta_cov(self, x_dot, eta):
        """ Updates the covariance matrix for the derivative

        Parameters
        ----------
        x_dot: numpy.ndarray
            The current centered sample derivative
        x_dot_prev: numpy.ndarray
            The previous centered sample derivative
        eta: float
            The learning rate

        Returns
        -------
        cov_delta: numpy.ndarray
            The estimated covariance matrix of the derivative of the samples
        """
        history = (1 - eta) * self.covariance_delta
        update = eta * (x_dot @ x_dot.T)
        cov_delta = history + update
        self.covariance_delta = cov_delta

        return(cov_delta)

    def _update_transformation_matrix(self, x_dot, eta, Q, svd=False):
        """ Updates the estimate of the transformation matrix

        Parameters
        ----------
        x_dot: numpy.ndarray
            The current centered sample derivative
        eta: float
            The learning rate
        Q: numpy.ndarray
            The current estimate of the whitening matrix
        svd: boolean
            Use direct SVD rather than QR Orthogonal Iteration Method

        Returns
        -------
        P: numpy.ndarray
            The matrix used to transform whitened data into the features.
        """
        cov_delta = self._update_delta_cov(x_dot, eta)
        if svd:
            PT, L, P = LA.svd(Q @ cov_delta @ Q.T, hermitian=True)
        else:
            gam = 4
            P = self.transformation_matrix
            S = np.eye(self.output_variables) - (Q @ cov_delta @ Q.T) / gam
            PT, R = LA.qr(S @ P.T)
            P = PT.T
        self._check_convergence(self.transformation_matrix, P)
        self.transformation_matrix = P
        return(P)

    def _check_convergence(self, P_prev, P, eta=0.05):
        """ Checks model convergence based on change in P

        Parameters
        ----------
        P_prev: numpy.ndarray
            The previous transformation matrix
        P: numpy.ndarray
            The updated transformation matrix
        eta: float
            The largest relative change indicating convergence
        """
        P = P[:, :self.Md]
        P_prev = P_prev[:, :self.Md]
        change = LA.norm(P - P_prev, ord='fro') / LA.norm(P, ord='fro')
        if change < eta:
            self.evaluation = True
            print(f"Transformation matrix has converged at time {self.time}")
        return

    def update_control_limits(self, features, features_delta,
                              cov_delta, P, Q, alpha):
        """ Get the new test statistics and critical values

        Parameters
        ----------
        features: numpy.ndarray
            The current feature output
        features_delta: numpy.ndarray
            The derivative estimate of the current feature output
        cov_delta: numpy.ndarray
            The estimated covariance matrix of the derivative of the samples
        P: numpy.ndarray
            The current estimate of the transformation matrix
        Q: numpy.ndarray
            The current estimate of the whitening matrix
        alpha: float
            The confidence value for the critical monitoring statistics

        Returns
        -------
        stats: array
            An array containing the monitoring statistics in the following
            order: T^2 (for slowest features), T^2 (for fastest features),
            S^2 (for slowest features)
        stats_crit: array
            An array containing the critical statistics in the same order as
            stats.
        """
        # Calculate theta values
        t1 = 0
        t2 = 0
        t3 = 0
        Om = np.zeros(self.Md)
        for i in range(self.Md):
            p = P[:, i].reshape(self.num_features, 1)
            w = p.T @ Q @ cov_delta @ Q.T @ p
            # TODO: Check if its to the power of j or something else?
            t1 += w
            t2 += w ** 2
            t3 += w ** 3
            Om[i] = w
        c = ST.norm.cdf(alpha)

        # Calculate critical Q statistic
        h = 1 - (2 * t1 * t3) / (3 * t2 * t2)
        q1 = c * (2 * t2 * h * h) ** (1/2)
        q2 = t2 * h * (h - 1) / (t1 * t1)
        Q = t1 * (q1 + q2 + 1) ** (1 / h)
        # Calculate critical T stats
        T_d_crit = ST.chi2.ppf(1 - alpha, self.Md)
        T_e_crit = ST.chi2.ppf(1 - alpha, self.Me)

        # Calculate test stats
        features_slow = features[:self.Md].reshape(self.Md, 1)
        features_fast = features[self.Md:].reshape(self.Me, 1)
        T_d = features_slow.T @ features_slow
        T_e = features_fast.T @ features_fast
        features_slow_delta = features_delta[:self.Md].reshape(self.Md, 1)
        S_d = features_slow_delta.T @ np.diag(Om ** -1) @ features_slow_delta

        stats = [T_d, T_e, S_d]
        stats_crit = [T_d_crit, T_e_crit, Q]
        return(stats, stats_crit)

    def add_data(self, raw_sample, alpha=0.05):
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
        if self.evaluation:
            y, stats, stats_crit = self._evaluate(raw_sample, alpha)
            return(y, stats, stats_crit)
        # Default values
        y = np.zeros((self.num_features, 1))
        stats = [0, 0, 0]
        stats_crit = [0, 0, 0]

        """ Signal preprocessing """
        raw_sample = raw_sample.reshape((raw_sample.shape[0], 1))
        sample = self.process_sample(raw_sample)
        if np.allclose(sample, 0):
            # Need more data to get right number of dynamic copies
            return(y, stats, stats_crit)

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
        """ Transformation """
        if self.time > 0:
            # Update derivatives
            x_dot_prev = np.copy(self.centered_delta_previous)
            x_dot = (x - x_prev)/self.delta
            self.centered_delta_current = x_dot
            self.centered_delta_previous = x_dot_prev

            # Transform data
            P = self._update_transformation_matrix(x_dot, eta, Q)
            y = P @ Q @ x

            # Update stored features
            self.feature_previous = np.copy(self.feature_current)
            y_prev = self.feature_previous
            self.feature_current = y
            if self.time > 1:
                y_delta = (y - y_prev) / self.delta
                """ Update monitoring stats """
                cov_delta = self.covariance_delta
                stats, stats_crit = self.update_control_limits(y, y_delta,
                                                               cov_delta,
                                                               P, Q, alpha)

        """ Update model and return """
        self.time += 1

        return(y, stats, stats_crit)

    def _evaluate(self, raw_sample, alpha=0.05):
        """Calculate the features and monitors without updating the model

        Parameters
        ----------
        raw_sample: numpy.ndarray
            The sample used to calculate the features

        Returns
        -------
        y: numpy.ndarray
            The calculated features for the input sample
        stats: array
            An array containing the monitoring statistics in the following
            order: T^2 (for slowest features), T^2 (for fastest features),
            S^2 (for slowest features).
        """
        """ Signal preprocessing """
        raw_sample = raw_sample.reshape((raw_sample.shape[0], 1))
        sample = self.process_sample(raw_sample)
        x = self.standardization_node.center_similar(sample)

        """ Transformation """
        Q = self.standardization_node.whitening_matrix
        P = self.transformation_matrix
        y = P @ Q @ x

        """ Update history """
        self.feature_previous = np.copy(self.feature_current)
        y_prev = self.feature_previous
        self.feature_current = y
        y_delta = (y - y_prev) / self.delta

        """ Get statistics """
        cov_delta = self.covariance_delta
        stats, stats_crit = self.update_control_limits(y, y_delta, cov_delta,
                                                       P, Q, alpha)

        """ Check for faults """
        self.consecutive_faults = self._check_faults(stats, stats_crit,
                                                     self.consecutive_faults,
                                                     self.required_faults)
        self.time += 1
        return(y, stats, stats_crit)

    def _check_faults(self, stats, stats_crit,
                      current_faults, required_faults):
        T_d, T_e, S_d = stats
        T_d_crit, T_e_crit, S_d_crit = stats_crit
        if T_d > T_d_crit or T_e > T_e_crit:
            if False:  # S_d > S_d_crit:
                print(f"Fault detected at time {self.time}, model updating")
                self.evaluation = False
            else:
                current_faults += 1
                if current_faults >= required_faults:
                    print(f"{current_faults} consecutive faults detected "
                          f"from time {self.time - current_faults} to "
                          f"{self.time}, model updating")
                    current_faults = 0
                    self.evaluation = False
        else:
            current_faults = 0
        return(current_faults)
