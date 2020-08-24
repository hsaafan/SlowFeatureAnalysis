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

import warnings
import itertools

import numpy as np
import scipy.linalg as LA
from scipy import stats
import matplotlib.pyplot as plt

import tep_import as imp
from data_node import IncrementalNode
from standardization_node import IncrementalStandardization

eps = 1e-64


class IncSFA(IncrementalNode):
    """ Incremental Slow Feature Analysis

    Incremental Slow feature analysis class that takes in data samples
    sequentially and updates the transformation matrix estimate and slow
    features. Extends the IncrementalNode class in data_node.py.

    Attributes
    ----------
    time: int
        The current time index which increments by 1 after each iteration
    num_features: int
        The amount of features to calculate.
    num_components: int
        The amount of principal components to calculate for data whitening.
    transformation_matrix: numpy.ndarray
        The matrix that transforms whitened data into the features.
    delta: float
        The time between samples in the data, default is 1
    standardization_node: IncrementalStandardization
        Object used to standardize input samples
    ccipa_object: IncrementalStandardization
        Object used to find the first principal component in whitened
        difference space
    standardized_previous: numpy.ndarray
        The previous sample that has been standardized
    standardized_current: numpy.ndarray
        The current sample that has been standardized
    feature_previous: numpy.ndarray
        The previous feature output
    feature_current: numpy.ndarray
        The current feature output
    _v_gam: numpy.ndarray
        The first principal component in whitened difference space. Used as an
        input for the minor component analysis algorithm.
    features_speed: numpy.ndarray
        The estimated singular values of the transformation matrix. These
        values are proportional to the speeds of the features.
    Md: int
        The amount of features that are considered "slow"
    Me: int
        The amount of features that are considered "fast"
    """
    time = 0
    num_features = 0
    num_components = 0
    transformation_matrix = None
    delta = 1
    standardized_previous = None
    standardized_current = None
    feature_previous = None
    feature_current = None
    _v_gam = None
    features_speed = None
    Md = None
    Me = None

    def __init__(self, input_variables, num_features, num_components,
                 theta=None, expansion_order=None, dynamic_copies=None):
        """ Class constructor

        Parameters
        ----------
        input_variables: int
            The amount of variables in the data before any dynamization or
            expansion.
        num_features: int
            The amount of features to calculate.
        num_components: int
            The amount of principal components to calculate for data whitening.
            Calculating more components will ensure that more information is
            kept from the original data but the more components used, the
            longer it will take for the model to converge.
        theta: array
            An array containing the scheduling parameters for setting the
            learning rate. See the _learning_rate_schedule for more
            information about these parameters.
        expansion_order: int
            The order of nonlinear expansion to perform on the data.
        dynamic_copies: int
            The amount of lagged copies to add to the data.
        """
        super().__init__(input_variables, dynamic_copies, expansion_order)
        self._store_theta(theta)

        J = num_features
        K = num_components

        self.num_components = K
        self.num_features = J
        self.transformation_matrix = np.eye(K, J)
        self._v_gam = np.random.randn(K, 1)

        self.features_speed = np.zeros(J)

    def _store_theta(self, theta):
        """ Checks and stores the scheduling parameters

        Parameters
        ----------
        theta: array
            The 7 scheduling parameters stored in an array. See the
            _learning_rate_schedule method for more information on these
            parameters.
        """
        if theta is None:
            warnings.warn("No scheduling parameters specified, using "
                          "example values: t1 = 20, t2 = 200, c = 3, "
                          "r = 2000, eta_l = 0, eta_h = 0.01", RuntimeWarning)
            self.theta = [20, 200, 3, 2000, 0, 0.01, 2000]
        elif len(theta) == 7:
            if not(theta[0] < theta[1]):
                raise RuntimeError("Values must be t1 < t2")
            if not(theta[4] <= theta[5]):
                raise RuntimeError("Values must be eta_l <= eta_h")
            self.theta = theta
        else:
            raise RuntimeError("Expected 7 parameters in order: " +
                               "t1, t2, c, r, eta_l, eta_h, T")
        return

    def _CIMCA_update(self, W, z_dot, gamma, eta, cov_free=True):
        """ Candid Covariance Free Incremental Minor Component Analysis

            Performs incremental minor component analysis [1]_.

        Parameters
        ----------
        W: numpy.ndarray
            The current estimate of the transformation matrix.
        z_dot: numpy.ndarray
            The derivative of the current whitened sample.
        gamma: numpy.ndarray
            The first principal component of the transformation matrix.

        Returns
        -------
        W: numpy.ndarray
            The new estimate of the transformation matrix.

        References
        ----------
        .. [1] Kompella, V., Luciw, M., & Schmidhuber, J. (2012). Incremental
        Slow Feature Analysis: Adaptive Low-Complexity Slow Feature Updating
        from High-Dimensional Input Streams. Neural Computation, 24(11),
        2994–3024. <https://doi.org/10.1162/neco_a_00344>
        """
        K, J = W.shape
        x = z_dot.reshape((-1, 1))
        if not cov_free:
            C = np.zeros((J, J))
            for i in range(J):
                wi = W[:, i].reshape((-1, 1))
                xxc = x @ x.T + C
                wxxcw = wi.T @ xxc @ wi
                delta = xxc @ wi - (wxxcw / (wi.T @ wi) ** 2) * wi
                wi = wi - eta * delta
                wi = wi / LA.norm(wi)
                W[:, i] = wi.flat
                C += gamma * (wi @ wi.T)
        else:
            L = np.zeros((K, 1))
            for i in range(J):
                lower_sum = np.zeros((K, 1))
                w_prev = W[:, i].reshape((-1, 1))
                for j in range(i):
                    wj = W[:, j].reshape((-1, 1))
                    lower_sum += (wj.T @ w_prev) * wj
                L = gamma * lower_sum

                wx = x.T @ w_prev
                wl = w_prev.T @ L
                a = (w_prev.T @ w_prev + eps) ** -2
                b = wx ** 2
                delta = wx * x + L - a*b*w_prev - a*wl*w_prev
                w_new = w_prev - eta * delta
                w_norm = LA.norm(w_new) + eps
                w_new = w_new / w_norm
                W[:, i] = w_new.flat
        return(W)

    def update_feature_speeds(self, Lambda, y_dot, eta=0.01):
        """ Estimate the eigenvalues of the transformation matrix

        Parameters
        ----------
        Lambda: numpy.ndarray
            Previous estimate of singular values of the transformation matrix
        y_dot: numpy.ndarray
            Estimated derivative of slow features
        eta: float
            Learning rate

        Returns
        -------
        Lambda: numpy.ndarray
            The new estimate of the singular values of the transformation
            matrix.
        """
        lam = np.multiply(y_dot, y_dot).reshape((-1))
        Lambda = (1 - eta) * Lambda + eta * lam
        return(Lambda)

    def calculate_monitoring_stats(self, Lambda, y, y_dot):
        """ Calculate the monitoring statistics

        Parameters
        ----------
        Lambda: numpy.ndarray
            Estimated singular values of the transformation matrix
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

        # Split features, derivatives, and speeds into slowest and fastest
        y_slow = y[:self.Md].reshape((-1, 1))
        y_fast = y[self.Md:].reshape((-1, 1))

        y_dot_slow = y_dot[:self.Md].reshape((-1, 1))
        y_dot_fast = y_dot[self.Md:].reshape((-1, 1))

        speed_slow = (Lambda[:self.Md]).reshape((-1))
        speed_fast = (Lambda[self.Md:]).reshape((-1))

        # Calculate monitoring stats
        T_d = y_slow.T @ y_slow
        T_e = y_fast.T @ y_fast

        S_d = y_dot_slow.T @ np.diag(speed_slow**-1) @ y_dot_slow
        S_e = y_dot_fast.T @ np.diag(speed_fast**-1) @ y_dot_fast

        return(T_d, T_e, S_d, S_e)

    def add_data(self, raw_sample, update_monitors=True,
                 calculate_monitors=False, use_svd_whitening=True):
        """ Update the IncSFA model with new data points

        Parameters
        ----------
        raw_sample: numpy.ndarray
            The sample used to update the model and calculate the features
        update_monitors: boolean
            Whether to update the slow features or not
        calculate_monitors: boolean
            Whether to calculate the monitoring statistics or not

        Returns
        -------
        y: numpy.ndarray
            The calculated features for the input sample
        stats: array
            An array containing the monitoring statistics in the following
            order: T^2 (for slowest features), T^2 (for fastest features),
            S^2 (for slowest features), S^2 (for fastest features). If
            calculate_monitors is set to False, these will all be 0.
        """
        # Default values
        y = np.zeros((self.num_features, 1))
        stats = [0, 0, 0, 0]

        """ Signal preprocessing """
        raw_sample = raw_sample.reshape((-1, 1))
        sample = self.process_sample(raw_sample)
        if np.allclose(sample, 0):
            # Need more data to get right number of dynamic copies
            return(y, stats)

        """ Update Learning rates """
        eta_PCA, eta_MCA = self._learning_rate_schedule(self.theta, self.time)

        """ Signal centering and whitening """
        if self.time > 0:
            # Updates normalized signals stored
            self.standardized_previous = np.copy(self.standardized_current)
            z_prev = self.standardized_previous
        else:
            # On first pass through, create the norm object
            node = IncrementalStandardization(sample, self.num_components)
            self.standardization_node = node
            self.cov = np.zeros((sample.shape[0], sample.shape[0]))
        if use_svd_whitening:
            self.cov = (1-eta_PCA) * self.cov + eta_PCA * sample @ sample.T
            U, L, UT = LA.svd(self.cov)
            Q = U @ np.diag(L ** -0.5)
            z = Q.T @ sample
            self.standardization_node.whitening_matrix = Q.T
        else:
            z = self.standardization_node.standardize_online(sample, eta_PCA)

        self.standardized_current = z
        """ Transformation """
        # time > 0, derivative can be calculated, proceed to update model
        delay = 0
        if self.time > 0 + delay:
            eta_PCA, eta_MCA = self._learning_rate_schedule(self.theta,
                                                            self.time - delay)
            z_dot = (z - z_prev)/self.delta
            # Update first PC in whitened difference space and gamma
            """ Derivative centering and first eigenvector output """
            if self.time == 1 + delay:
                self.ccipa_object = IncrementalStandardization(z_dot, 1)
            self._v_gam = self.ccipa_object.update_CCIPA(z_dot, eta_PCA)
            # gam = self._v_gam / (LA.norm(self._v_gam) + 1e-64)
            gam = LA.norm(self._v_gam)
            # Updates minor components
            # gam = 4
            # z_dot = self.ccipa_object._center(z_dot, eta_PCA)
            P = self.transformation_matrix
            P = self._CIMCA_update(P, z_dot, gam, eta_PCA, cov_free=True)
            # P = FDPM(P, z_dot, mu_bar=0.95)
            self.transformation_matrix = P

            y = P.T @ z

            self.feature_previous = np.copy(self.feature_current)
            y_prev = self.feature_previous
            self.feature_current = y

            """ Update monitoring stats """
            if self.time > 2 + delay:
                # Approximate derivative
                y_dot = (y - y_prev)/self.delta
                if update_monitors:
                    # Update the singular value estimate
                    Lam = self.features_speed
                    Lam = self.update_feature_speeds(Lam, y_dot, eta=0.01)
                    self.features_speed = Lam
                if calculate_monitors:
                    # Calculate the monitoring stats
                    Lam = self.features_speed
                    stats = self.calculate_monitoring_stats(Lam, y, y_dot,
                                                            self.Md)
        """ Update model and return """
        self.time += 1
        return(y, stats)

    def evaluate(self, raw_sample, alpha):
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
            An array containing the critical statistics in the same order as
            stats.
        """
        # Default values
        y = np.zeros((self.num_features, 1))
        stats = [0, 0, 0, 0]
        stats_crit = [0, 0, 0, 0]

        """ Signal preprocessing """
        raw_sample = raw_sample.reshape((raw_sample.shape[0], 1))
        sample = self.process_sample(raw_sample)
        if np.allclose(sample, 0):
            raise RuntimeError("Not enough data has been passed to the model "
                               "to test it")

        """ Signal centering and whitening """
        self.standardized_previous = np.copy(self.standardized_current)
        z = self.standardization_node.standardize_similar(sample)
        self.standardized_current = z

        """ Transformation """
        y_prev = np.copy(self.feature_current)
        self.feature_previous = y_prev
        P = self.transformation_matrix
        y = P.T @ z
        self.feature_current = y

        # Approximate derivative
        y_dot = (y - y_prev)/self.delta

        # Calculate the monitoring stats
        stats = self.calculate_monitoring_stats(self.features_speed, y, y_dot)
        stats_crit = self.calculate_crit_values(alpha)

        return(y, stats, stats_crit)

    def _learning_rate_schedule(self, theta, time):
        """ Sets the learning rates

        Parameters
        ----------
        theta: array
            The array containing the scheduling parameters
        time: int
            The current time index

        Returns
        -------
        eta_PCA: float
            The learning rate for principal component analysis
        eta_MCA: float
            The learning rate for minor component analysis
        """
        t1, t2, c, r, nl, nh, T = theta
        time += 1  # Prevents divide by 0 errors
        """
        if time == 1:
            mu_t = -1e-6
        elif time <= t1:
            mu_t = 0
        elif time <= t2:
            mu_t = c * (time - t1)/(t2 - t1)
        else:
            mu_t = c + (time - t2)/r
        eta_PCA = (1 + mu_t)/time
        """

        if time <= T:
            eta_MCA = nl + (nh - nl)*((time / T)**2)
        else:
            eta_MCA = nh
        L = 2
        eta_PCA = np.max([(1 + L)/time, 1e-4])
        return(eta_PCA, eta_MCA)

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
        if alpha > 1:
            raise ValueError("Confidence level is capped at 1")
        p = 1 - alpha
        N = self.time
        Md = self.Md
        Me = self.Me
        gd = (Md*(N**2-2*N))/((N-1)*(N-Md-1))
        ge = (Me*(N**2-2*N))/((N-1)*(N-Me-1))

        T_d_crit = stats.chi2.ppf(p, Md)
        T_e_crit = stats.chi2.ppf(p, Me)
        S_d_crit = gd*stats.f.ppf(p, Md, N-Md-1)
        S_e_crit = ge*stats.f.ppf(p, Me, N-Me-1)

        crit_vals = [T_d_crit, T_e_crit, S_d_crit, S_e_crit]

        return(crit_vals)
