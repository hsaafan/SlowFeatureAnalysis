""" Incremental Slow Feature Analysis

Implementation of Incremental Slow Feature Analysis [1]_

Classes
-------
IncSFA: class
    The class used for Incremental Slow Feature Analysis

References
----------
.. [1] Kompella, V., Luciw, M., & Schmidhuber, J. (2012). Incremental Slow
       Feature Analysis: Adaptive Low-Complexity Slow Feature Updating from
       High-Dimensional Input Streams. Neural Computation, 24(11), 2994–3024.
       <https://doi.org/10.1162/neco_a_00344>
"""
__author__ = "Hussein Saafan"

import numpy as np
import scipy.linalg as LA
import scipy.stats as ST

import tep_import as imp
from data_node import IncrementalNode
from standardization_node import IncrementalStandardization
from standardization_node import RecursiveStandardization

eps = 1e-64


class IncSFA(IncrementalNode):
    """ Incremental Slow Feature Analysis

    Incremental Slow feature analysis class that takes in data samples
    sequentially and updates the transformation matrix estimate and slow
    features. Extends the IncrementalNode class in data_node.py.

    Attributes
    ----------
    converged: boolean
        Whether the model has converged yet
    time: int
        The current time index which increments by 1 after each iteration
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
    standardized_sample: numpy.ndarray
        The current sample that has been standardized
    feature_sample: numpy.ndarray
        The current feature output
    features_speed: numpy.ndarray
        The estimated singular values of the transformation matrix. These
        values are proportional to the speeds of the features
    standardization_node: IncrementalStandardization/RecursiveStandardization
        Object used to standardize input samples
    ccipa_object: IncrementalStandardization
        Object used to find the first principal component in whitened
        difference space
    """
    converged = False

    time = 0
    num_components = 0
    num_features = 0
    Md = 0
    Me = 0

    delta = 1.0
    L = 0.0
    conv_tol = 0.0

    transformation_matrix = None
    standardized_sample = None
    feature_sample = None
    features_speed = None

    standardization_node = None
    ccipa_object = None

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

        self.num_components = K
        self.num_features = J
        self.L = L
        self.conv_tol = conv_tol
        self.transformation_matrix = np.eye(K, J)
        self.features_speed = np.zeros(J)

    def _CIMCA_update(self, P, z_dot, gamma, eta):
        """ Candid Covariance Free Incremental Minor Component Analysis

            Performs incremental minor component analysis [1]_.

        Parameters
        ----------
        P: numpy.ndarray
            The current estimate of the transformation matrix.
        z_dot: numpy.ndarray
            The derivative of the current whitened sample.
        gamma: numpy.ndarray
            The first principal component of the transformation matrix.
        eta: float
            The learning rate

        Returns
        -------
        P: numpy.ndarray
            The new estimate of the transformation matrix.

        References
        ----------
        TODO: Change this reference
        .. [1] Kompella, V., Luciw, M., & Schmidhuber, J. (2012). Incremental
        Slow Feature Analysis: Adaptive Low-Complexity Slow Feature Updating
        from High-Dimensional Input Streams. Neural Computation, 24(11),
        2994–3024. <https://doi.org/10.1162/neco_a_00344>
        """
        K = self.num_components
        J = self.num_features
        x = z_dot.reshape((-1, 1))
        L = np.zeros((K, 1))
        lower_sum = np.zeros((K, 1))
        for i in range(J):
            p_prev = P[:, i].reshape((-1, 1))
            coefficients = P[:, :i].T @ p_prev
            if i > 0:
                lower_sum = np.sum(np.multiply(P[:, :i], coefficients.T),axis=1)
            L = (gamma * lower_sum).reshape((K, 1))

            p_new = (1-eta) * p_prev - eta*((z_dot.T @ p_prev)*z_dot + L)
            p_norm = LA.norm(p_new) + eps
            p_new = p_new / p_norm
            P[:, i] = p_new.flat
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

    def _update_feature_speeds(self, Omega, y_dot, eta=0.01):
        """ Estimate the eigenvalues of the transformation matrix

        Parameters
        ----------
        Omega: numpy.ndarray
            Previous estimate of singular values of the transformation matrix
        y_dot: numpy.ndarray
            Estimated derivative of slow features
        eta: float
            Learning rate

        Returns
        -------
        Omega: numpy.ndarray
            The new estimate of the singular values of the transformation
            matrix.
        """
        new_vals = np.multiply(y_dot, y_dot).reshape((-1))
        Omega = (1 - eta) * Omega + eta * new_vals
        return(Omega)

    def add_data(self, raw_sample, alpha=0.01, update_monitors=True,
                 calculate_monitors=False, use_svd_whitening=True):
        """ Update the IncSFA model with new data points

        Parameters
        ----------
        raw_sample: numpy.ndarray
            The sample used to update the model and calculate the features
        alpha: float
            The significance level of the monitoring stats
        update_monitors: boolean
            Whether to update the slow features or not
        calculate_monitors: boolean
            Whether to calculate the monitoring statistics or not
        use_svd_whitening: boolean
            Use rank-one updating of covariance matrix and SVD rather than
            the CCIPCA method

        Returns
        -------
        y: numpy.ndarray
            The calculated features for the input sample
        stats: array
            An array containing the monitoring statistics in the following
            order: T^2 (for slowest features), T^2 (for fastest features),
            S^2 (for slowest features), S^2 (for fastest features). If
            calculate_monitors is set to False, these will all be 0.
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

        """ Update Learning rates """
        eta = self._learning_rate_schedule(self.L, self.time)

        """ Signal centering and whitening """
        if self.time > 0:
            z_prev = np.copy(self.standardized_sample)
        else:
            # On first pass through, create the norm object
            if use_svd_whitening:
                node = RecursiveStandardization(sample, self.num_components)
                self.standardization_node = node
            else:
                node = IncrementalStandardization(sample, self.num_components)
                self.standardization_node = node

        z = self.standardization_node.standardize_online(sample, eta)
        self.standardized_sample = z

        """ Transformation """
        if self.time > 0:
            z_dot = (z - z_prev)/self.delta
            # Update first PC in whitened difference space and gamma
            gam = 4
            """ Update feature transform matrix """
            P = self._CIMCA_update(self.transformation_matrix, z_dot, gam, eta)
            self.transformation_matrix = P

            """ Calculate features """
            y = P.T @ z

            y_prev = np.copy(self.feature_sample)
            self.feature_sample = y

            """ Update monitoring stats """
            if self.time > 1:
                # Approximate derivative
                y_dot = (y - y_prev)/self.delta
                if update_monitors:
                    # Update the singular value estimate
                    Omega = self.features_speed
                    Omega = self._update_feature_speeds(Omega, y_dot, eta)
                    self.features_speed = Omega
                if calculate_monitors:
                    # Calculate the monitoring stats
                    Omega = self.features_speed
                    stats = self.calculate_monitors(Omega, y, y_dot)
                    stats_crit = self.calculate_crit_values(alpha)
        """ Update model and return """
        self.time += 1
        return(y, stats, stats_crit)

    def _evaluate(self, raw_sample, alpha):
        """ Test the model without updating it with new data points

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
            S^2 (for slowest features), S^2 (for fastest features)
        stats_crit: array
            The critical monitoring statistics in the same order as stats
        """

        """ Signal preprocessing """
        raw_sample = raw_sample.reshape((raw_sample.shape[0], 1))
        sample = self.process_sample(raw_sample)
        if np.allclose(sample, 0):
            raise RuntimeError("Not enough data has been passed to the model "
                               "to test it")

        """ Signal centering and whitening """
        z_prev = np.copy(self.standardized_sample)
        z = self.standardization_node.standardize_similar(sample)
        self.standardized_sample = z

        """ Transformation """
        y_prev = np.copy(self.feature_sample)
        P = self.transformation_matrix
        y = P.T @ z
        self.feature_sample = y

        # Approximate derivative
        y_dot = (y - y_prev)/self.delta

        # Calculate the monitoring stats
        stats = self.calculate_monitors(self.features_speed, y, y_dot)
        stats_crit = self.calculate_crit_values(alpha)

        return(y, stats, stats_crit)

    def _learning_rate_schedule(self, L, time):
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
        time += 1  # Prevent divby0 errors
        eta = np.max([(1 + L)/time, 1e-4])
        return(eta)

    def calculate_crit_values(self, alpha=0.01):
        """ Calculate critical values for monitoring

        Parameters
        ----------
        alpha: float
            The confidence level to use for calculating the critical monitoring
            statistics.

        Returns
        -------
        crit_vals: Array
            An array containing the followin values:
            T_d_crit:
                The critical T^2 value for the slowest features
            T_e_crit: float
                The critical T^2 value for the fastest features
            S_d_crit: float
                The critical S^2 value for the slowest features
            S_e_crit: float
                The critical S^2 value for the fastest features
        """
        if alpha > 1 or alpha < 0:
            raise ValueError("Confidence level should be between 0 and 1")
        p = 1 - alpha
        n = self.time
        Md = self.Md
        Me = self.Me
        gd = (Md*(n**2-2*n))/((n-1)*(n-Md-1))
        ge = (Me*(n**2-2*n))/((n-1)*(n-Me-1))

        T_d_crit = ST.chi2.ppf(p, Md)
        T_e_crit = ST.chi2.ppf(p, Me)
        S_d_crit = gd*ST.f.ppf(p, Md, n-Md-1)
        S_e_crit = ge*ST.f.ppf(p, Me, n-Me-1)

        crit_vals = [T_d_crit, T_e_crit, S_d_crit, S_e_crit]

        return(crit_vals)

    def calculate_monitors(self, Omega, y, y_dot):
        """ Calculate the monitoring statistics

        Parameters
        ----------
        Omega: numpy.ndarray
            Estimated eigenvalues of the transformation matrix
        y: numpy.ndarray
            Current sample features
        y_dot: numpy.ndarray
            Current sample feature derivative

        Returns
        -------
        T_d: float
            The T^2 statistic for the slowest features
        T_e: float
            The T^2 statistic for the fastest features
        S_d: float
            The S^2 statistic for the slowest features
        S_e: float
            The S^2 statistic for the fastest features
        """
        y = y.reshape((-1))
        y_dot = y_dot.reshape((-1))

        # Order slow features in order of increasing speed
        order = Omega.argsort()
        Omega = Omega[order]
        y = y[order]
        y_dot = y_dot[order]

        # Split features, derivatives, and speeds into slowest and fastest
        y_slow = y[:self.Md].reshape((-1, 1))
        y_fast = y[self.Md:].reshape((-1, 1))

        y_dot_slow = y_dot[:self.Md].reshape((-1, 1))
        y_dot_fast = y_dot[self.Md:].reshape((-1, 1))

        speed_slow = (Omega[:self.Md]).reshape((-1))
        speed_fast = (Omega[self.Md:]).reshape((-1))

        # Calculate monitoring stats
        T_d = float(y_slow.T @ y_slow)
        T_e = float(y_fast.T @ y_fast)

        S_d = float(y_dot_slow.T @ np.diag(speed_slow**-1) @ y_dot_slow)
        S_e = float(y_dot_fast.T @ np.diag(speed_fast**-1) @ y_dot_fast)

        return(T_d, T_e, S_d, S_e)
