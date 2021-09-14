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
import scipy.linalg as LA
import scipy.stats as ST

from data_node import IncrementalNode
from standardization_node import RecursiveStandardization


class RSFA(IncrementalNode):
    """ Recursive Slow Feature Analysis

    Recursive Slow feature analysis class that takes in data samples
    sequentially and updates the transformation matrix estimate and slow
    features. Extends the IncrementalNode class in data_node.py.

    Attributes
    ----------
    converged: boolean
        Whether the model has converged yet
    time: int
        The current time index which increments by 1 after each iteration
    consecutive_faults: int
        The current number of faults that have been detected
    required_faults: int
        The number of faults required for the model to update
    num_components: int
        The amount of principal components to calculate for data whitening
    num_features: int
        The amount of features to calculate
    Md: int
        The amount of features that are considered "slow"
    Me: int
        The amount of features that are considered "fast"
    delta: float
        The time between samples in the data, default is 1
    L: float
        The sample weighting factor for the learning rate
    conv_tol: float
        The tolerance for convergence
    transformation_matrix: numpy.ndarray
        The matrix that transforms whitened data into the features
    centered_sample: numpy.ndarray
        The current sample that has been centered
    feature_sample: numpy.ndarray
        The current feature output
    features_speed: numpy.ndarray
        The singular values of the transformation matrix. These
        values are proportional to the speeds of the features
    covariance_delta: numpy.ndarray
        The estimated covariance matrix of the derivative of the samples
    standardization_node: RecursiveStandardization
        Object used to standardize input samples
    """
    converged = False

    time = 0
    num_components = 0
    num_features = 0
    consecutive_faults = 0
    required_faults = 25
    Md = 0
    Me = 0

    delta = 1.0
    L = 0.0
    conv_tol = 0.0

    transformation_matrix = None
    centered_sample = None
    feature_sample = None
    covariance_delta = None

    standardization_node = None

    def __init__(self, input_variables, num_features, num_components,
                 L=0, expansion_order=None, dynamic_copies=None,
                 conv_tol=0.01):
        """ Class constructor

        Parameters
        ----------
        input_variables: int
            The amount of variables in the data before any dynamization or
            expansion
        num_features: int
            The amount of features to calculate
        num_components: int
            The amount of principal components to calculate for data whitening.
            Calculating more components will ensure that more information is
            kept from the original data but the more components used, the
            longer it will take for the model to converge
        L: float
            The sample weighting parameter for the learning rate schedule
        expansion_order: int
            The order of nonlinear expansion to perform on the data
        dynamic_copies: int
            The amount of lagged copies to add to the data
        conv_tol: float
            The tolerance for convergence
        """
        super().__init__(input_variables, dynamic_copies, expansion_order)

        K = num_components
        J = num_features
        m = self.num_signals

        self.num_components = K
        self.num_features = J
        self.L = L
        self.conv_tol = conv_tol
        self.transformation_matrix = np.eye(K, J)
        self.features_speed = np.zeros(J)
        self.covariance_delta = np.zeros((m, m))

        self.update_after_converge = False

    def _learning_schedule(self, L, time):
        """ Sets the learning rate

        Parameters
        ----------
        L: float
            The sample weighting parameter
        time: int
            The current time index

        Returns
        -------
        eta: float
            The learning rate
        """
        time += 1
        eta = np.max([(1 + L) / time, 1e-4])
        return(eta)

    def _update_delta_cov(self, x_dot, eta):
        """ Updates the covariance matrix for the derivative

        Parameters
        ----------
        x_dot: numpy.ndarray
            The current centered sample derivative
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
            P, L, PT = LA.svd(Q.T @ cov_delta @ Q)
        else:
            gam = 4
            P = self.transformation_matrix
            S = np.eye(self.num_features) - (Q.T @ cov_delta @ Q) / gam
            P, _ = LA.qr(S @ P)
        # self._check_convergence(self.transformation_matrix, P)
        return(P)

    def _check_convergence(self, P_prev, P, ignore_resid=True):
        """ Checks model convergence based on change in P

        Parameters
        ----------
        P_prev: numpy.ndarray
            The previous transformation matrix
        P: numpy.ndarray
            The updated transformation matrix
        ignore_resid: boolean
            Only check the slowest features
        """
        if ignore_resid:
            P = P[:, :self.Md]
            P_prev = P_prev[:, :self.Md]
        change = LA.norm(P - P_prev, ord='fro') / LA.norm(P, ord='fro')
        if change < self.conv_tol:
            self.converged = True
            print(f"Transformation matrix has converged at time {self.time}")
        return

    def update_control_limits(self, y, y_dot, cov_X_dot, P, Q, alpha):
        """ Get the new test statistics and critical values

        Parameters
        ----------
        y: numpy.ndarray
            The current feature output
        y_dot: numpy.ndarray
            The derivative estimate of the current feature output
        cov_X_dot: numpy.ndarray
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
            S^2 (for slowest features), S^2 (for fastest features)
        stats_crit: array
            An array containing the critical statistics in the same order as
            stats.
        """
        # Calculate eigenvalues of transformation matrix
        Omega = P.T @ Q.T @ cov_X_dot @ Q @ P

        # Order slow features in order of increasing speed
        Omega = np.diag(Omega)
        order = Omega.argsort()
        Omega = Omega[order]
        y = y[order]
        y_dot = y_dot[order]
        self.features_speed = Omega

        # Calculate critical T stats
        Md = self.Md
        Me = self.num_features - self.Md

        T_d_crit = ST.chi2.ppf(1 - alpha, Md)
        T_e_crit = ST.chi2.ppf(1 - alpha, Me)
        Q_d_crit = self.calculate_Q_stat(Omega[:Md], alpha)
        Q_e_crit = self.calculate_Q_stat(Omega[Md:], alpha)

        # Calculate test stats
        features_slow = y[:Md].reshape(-1, 1)
        features_fast = y[Md:].reshape(-1, 1)
        T_d = features_slow.T @ features_slow
        T_e = features_fast.T @ features_fast

        features_slow_delta = y_dot[:Md].reshape(-1, 1)
        features_fast_delta = y_dot[Md:].reshape(-1, 1)

        # speed_slow = (Omega[:Md]).reshape((-1))
        # speed_fast = (Omega[Md:]).reshape((-1))

        # S_d = features_slow_delta.T @ np.diag(speed_slow**-1) @ features_slow_delta
        # S_e = features_fast_delta.T @ np.diag(speed_fast**-1) @ features_fast_delta

        S_d = features_slow_delta.T @ features_slow_delta
        S_e = features_fast_delta.T @ features_fast_delta

        stats = [T_d, T_e, S_d, S_e]
        stats_crit = [T_d_crit, T_e_crit, Q_d_crit, Q_e_crit]
        return(stats, stats_crit)

    def calculate_Q_stat(self, eigenvalues, alpha, use_chi2=False):
        """ Calculate the critical Q statistic

        Parameters
        ----------
        eigenvalues: numpy.ndarray
            A 1-D array containing the eigenvalues used for the calculation
        alpha: float
            The significance level to use
        use_chi2: boolean
            Whether to use the alternative scaled chi2 statistic instead
        """
        # Calculate theta values
        t1 = np.sum(eigenvalues)
        t2 = np.sum(eigenvalues ** 2)
        t3 = np.sum(eigenvalues ** 3)

        if use_chi2:
            # Alternative Q using chi squared
            g = t2 / t1
            h = t1 * t1 / t2
            Q = g * ST.chi2.ppf(1 - alpha, h)
        else:
            # Calculate Q stat
            c = ST.norm.ppf(1 - alpha)
            h = 1 - (2 * t1 * t3) / (3 * t2 * t2)
            q1 = (c * (2 * t2 * h * h) ** (1/2)) / t1
            q2 = t2 * h * (h - 1) / (t1 * t1)
            Q = t1 * (q1 + q2 + 1) ** (1 / h)
        return(Q)

    def add_data(self, raw_sample, alpha=0.01):
        """Update the RSFA model with new data points

        Parameters
        ----------
        raw_sample: numpy.ndarray
            The sample used to update the model and calculate the features
        alpha: float
            The significance level of the monitoring stats

        Returns
        -------
        y: numpy.ndarray
            The calculated features for the input sample
        stats: array
            An array containing the monitoring statistics in the following
            order: T^2 (for slowest features), T^2 (for fastest features),
            S^2 (for slowest features), and S^2 (for fastest features)
        stats_crit: array
            The critical monitoring statistics in the same order as stats
        """
        if self.converged:
            y, stats, stats_crit = self._evaluate(raw_sample, alpha)
            return(y, stats, stats_crit)
        # Default values
        y = np.zeros((self.num_features, 1))
        stats = [0, 0, 0, 0]
        stats_crit = [0, 0, 0, 0]

        """ Signal preprocessing """
        raw_sample = raw_sample.reshape((-1, 1))
        sample = self.process_sample(raw_sample)
        if np.allclose(sample, 0):
            # Need more data to get right number of dynamic copies
            return(y, stats, stats_crit)

        """ Update Learning rate """
        eta = self._learning_schedule(self.L, self.time)

        """ Signal centering and whitening """
        if self.time > 0:
            # Updates centered signals stored
            x_prev = np.copy(self.centered_sample)
        else:
            # On first pass through, create the standardization object
            node = RecursiveStandardization(sample, self.num_components)
            self.standardization_node = node

        """ Centering and updating whitening matrix """
        x = self.standardization_node._center(sample, eta)
        _ = self.standardization_node._whiten(x, eta)
        Q = self.standardization_node.whitening_matrix

        self.centered_sample = x

        """ Transformation """
        if self.time > 0:
            x_dot = (x - x_prev)/self.delta

            """ Update feature transform matrix """
            P = self._update_transformation_matrix(x_dot, eta, Q)
            self.transformation_matrix = P

            """ Calculate features """
            W = P.T @ Q.T
            y = W @ x

            y_prev = np.copy(self.feature_sample)
            self.feature_sample = y

            """ Update monitoring stats """
            if self.time > 1:
                y_dot = (y - y_prev) / self.delta
                """ Update monitoring stats """
                cov_delta = self.covariance_delta
                stats, stats_crit = self.update_control_limits(y, y_dot,
                                                               cov_delta,
                                                               P, Q, alpha)

        """ Update model and return """
        self.time += 1

        return(y, stats, stats_crit)

    def _evaluate(self, raw_sample, alpha):
        """Calculate the features and monitors without updating the model

        Parameters
        ----------
        raw_sample: numpy.ndarray
            The sample used to calculate the features
        alpha: float
            The confidence value for the critical monitoring statistics

        Returns
        -------
        y: numpy.ndarray
            The calculated features for the input sample
        stats: array
            An array containing the monitoring statistics in the following
            order: T^2 (for slowest features), T^2 (for fastest features),
            S^2 (for slowest features), and S^2 (for fastest features)
        stats_crit: array
            The critical monitoring statistics in the same order as stats
        """

        """ Signal preprocessing """
        raw_sample = raw_sample.reshape((raw_sample.shape[0], 1))
        sample = self.process_sample(raw_sample)
        if np.allclose(sample, 0):
            raise RuntimeError("Not enough data has been passed to the model "
                               "to test it")

        """ Signal centering """
        x_prev = np.copy(self.centered_sample)
        x = self.standardization_node.center_similar(sample)
        self.centered_sample = x

        """ Transformation """
        Q = self.standardization_node.whitening_matrix
        P = self.transformation_matrix
        W = P.T @ Q.T
        y = W @ x

        """ Update history """
        y_prev = np.copy(self.feature_sample)
        self.feature_sample = y
        y_dot = (y - y_prev) / self.delta

        """ Get statistics """
        cov_delta = self.covariance_delta
        stats, stats_crit = self.update_control_limits(y, y_dot, cov_delta,
                                                       P, Q, alpha)

        """ Check for faults """
        if self.update_after_converge:
            c_f = self.consecutive_faults
            r_f = self.required_faults
            c_f = self._check_faults(stats, stats_crit, c_f, r_f)
            self.consecutive_faults = c_f

        self.time += 1
        return(y, stats, stats_crit)

    def _check_faults(self, stats, stats_crit,
                      current_faults, required_faults):
        T_d, T_e, S_d, S_e = stats
        T_d_crit, T_e_crit, S_d_crit, S_d_crit = stats_crit
        if T_d > T_d_crit or T_e > T_e_crit:
            if S_d > S_d_crit:
                print("Operating condition deviation detected at time "
                      f"{self.time}, model updating")
                self.converged = False
            else:
                current_faults += 1
                if current_faults >= required_faults:
                    print(f"{current_faults} consecutive faults detected "
                          f"from time {self.time - current_faults} to "
                          f"{self.time}, model updating")
                    current_faults = 0
                    self.converged = False
        else:
            current_faults = 0
        return(current_faults)
