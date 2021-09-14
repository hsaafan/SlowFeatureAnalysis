""" Slow Feature Analysis

Implementation of Slow Feature Analysis [1].

References
----------
.. [1] Shang, C., Yang, F., Gao, X., Huang, X., Suykens, J., & Huang, D.
       (2015). Concurrent monitoring of operating condition deviations and
       process dynamics anomalies with slow feature analysis. AIChE Journal, 6
       (11), 3666–3682. <https://doi.org/10.1002/aic.14888>

"""
__author__ = "Hussein Saafan"

import numpy as np
import scipy.linalg as LA
import scipy.stats as stats

from .data_node import Node
from .standardization_node import Standardization


class SFA(Node):
    """ Slow Feature Analysis

    Slow feature analysis class that takes a set of input signals and
    calculates the transformation matrix and slow features for the inputs.
    Extends the Node class in data_node.py.

    Attributes
    ----------
    trained: boolean
        Has the model been trained yet
    parted: boolean
        Have the slow features been partitioned into slowest and fastest
    training_samples: int
        The number of samples the model was trained on
    Md: int
        The amount of features that are considered "slow"
    Me: int
        The amount of features that are considered "fast"
    delta: float
        The time between samples in the data, default is 1
    features: numpy.ndarray
        The learned features from the training data
    slow_features: numpy.ndarray
        The slowest learned features of the training data based on where the
        model was partitioned
    fast_features: numpy.ndarray
        The fastest learned features of the training data based on where the
        model was partitioned
    transformation_matrix: numpy.ndarray
        The matrix that transforms whitened data into the features
    training_data: numpy.ndarray
        The data used to train the model
    standardized_data: numpy.ndarray
        The data used to train the model that has been standardized
    features_speed: numpy.ndarray
        The singular values of the transformation matrix that were found using
        singular value decompisition. These values are proportional to the
        speeds of the features.
    standardization_node: Standardization
        Used to standardize training and online data
    """
    trained = False
    parted = False

    training_samples = 0
    Md = 0
    Me = 0

    delta = 1.0

    features = None
    transformation_matrix = None
    training_data = None
    standardized_data = None
    features_speed = None

    standardization_node = None

    @property
    def slow_features(self):
        if not self.parted:
            raise RuntimeError("Signals have not been partitioned yet")
        return(self.features[:self.Md, :])

    @property
    def fast_features(self):
        if not self.parted:
            raise RuntimeError("Signals have not been partitioned yet")
        return(self.features[self.Md:, :])

    def __init__(self, data, dynamic_copies=None, expansion_order=None):
        """ Class constructor

        Parameters
        ----------
        data: numpy.ndarray
            The data to use for training the model which must be of shape
            (m, n) where m is the number of variables and n is the number
            of samples
        dynamic_copies: int
            The amount of lagged copies to add to the data
        expansion_order: int
            The order of nonlinear expansion to perform on the data
        """
        self._check_input_data(data)
        m = data.shape[0]
        n = data.shape[1]
        super().__init__(m, dynamic_copies, expansion_order)
        self.training_samples = n
        self.training_data = self.process_data(data)

    def _standardize_training(self):
        """ Updates the standardization model using the training data """
        self.standardization_node = Standardization(self.training_data)
        self.standardized_data = self.standardization_node.standardize()
        return

    def _transform_training(self):
        """ Transforms whitened data into features and updates the model """
        # Approximate the first order time derivative of the signals
        Z = self.standardized_data
        Z_dot = np.diff(Z)/self.delta
        # SVD of the covariance matrix of whitened difference data
        cov_Z_dot = np.cov(Z_dot)
        P, Omega, PT = LA.svd(cov_Z_dot)
        # The loadings are ordered to find the slowest varying features
        Omega, P = self._order(Omega, P)
        self.features_speed = Omega
        # Calculate transformation matrix and features
        Q = self.standardization_node.whitening_matrix
        self.transformation_matrix = P.T @ Q.T
        self.features = P.T @ Z
        return

    def _order(self, eigenvalues, eigenvectors):
        """ Order eigensystem by ascending eigenvalues

        Parameters
        ----------
        eigenvalues: numpy.ndarray
            1-D array of eigenvalues
        eigenvectors: numpy.ndarray
            2-D array where each column is an eigenvector

        Returns
        -------
        eigenvalues: numpy.ndarray
            1-D array of sorted eigenvalues
        eigenvectors: numpy.ndarray
            2-D array of sorted eigenvectors
        """
        order = eigenvalues.argsort()
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        return(eigenvalues, eigenvectors)

    def train(self):
        """ Train the model and return the features from the training data

        Returns
        -------
        features: numpy.ndarray
            Learned features from training data
        """
        self._standardize_training()
        self._transform_training()
        self.trained = True
        return(self.features)

    def partition(self, q=0.1):
        """ Partitions features based on upper quantile of input signal speeds

        Parameters
        ----------
        q: float
            Decimal between 0 and 1 used to get the q upper quantile of the
            input signal speeds
        """
        if not self.trained:
            raise RuntimeError("Model has not been trained yet")

        Z = self.standardization_node.standardize_similar(self.training_data)
        m = self.num_signals
        n = self.training_samples
        # Find slowness of input signals
        Z_dot = np.diff(Z)/self.delta
        signal_speed = np.zeros(m)
        for i in range(m):
            signal_derivative = Z_dot[i, :].reshape(-1, 1)
            signal_speed[i] = (signal_derivative.T @ signal_derivative)/(n-1)

        # Find where to partition slow features
        threshold = np.quantile(signal_speed, 1-q, interpolation='lower')
        for i in range(self.features_speed.size):
            if self.features_speed[i] > threshold:
                self.Md = i
                self.Me = self.features_speed.size - i
                break
        self.parted = True
        self.calculate_crit_values()
        return

    def partition_manual(self, Md):
        """ Manually select where to make the cut from slow to fast features

        Parameters
        ----------
        Me: int
            The index of the slow feature where the cuttoff happens. Anything
            less than Me is considered slow and the remainder are considered
            fast.
        """
        self.Md = Md
        self.Me = self.features_speed.size - Md
        self.parted = True
        self.calculate_crit_values()
        return

    def calculate_crit_values(self, alpha=0.01):
        """ Calculate critical values for monitoring

        Parameters
        ----------
        alpha: float
            The confidence level to use for the critical values

        Returns
        -------
        T_d_crit: float
            The critical T^2 value for the slowest features
        T_e_crit: float
            The critical T^2 value for the fastest features
        S_d_crit: float
            The critical S^2 value for the slowest features
        S_e_crit: float
            The crtiical S^2 value for the fastest features
        """
        if not self.parted:
            raise RuntimeError("Signals have not been partitioned yet")
        if alpha > 1 or alpha < 0:
            raise ValueError("Confidence level should be between 0 and 1")
        p = 1 - alpha
        n = self.training_samples
        Md = self.Md
        Me = self.Me
        gd = (Md*(n**2-2*n))/((n-1)*(n-Md-1))
        ge = (Me*(n**2-2*n))/((n-1)*(n-Me-1))

        T_d_crit = stats.chi2.ppf(p, Md)
        T_e_crit = stats.chi2.ppf(p, Me)
        S_d_crit = gd*stats.f.ppf(p, Md, n-Md-1)
        S_e_crit = ge*stats.f.ppf(p, Me, n-Me-1)

        return(T_d_crit, T_e_crit, S_d_crit, S_e_crit)

    def calculate_monitors(self, online_data):
        """ Calculate monitoring statistics for test data

        Parameters
        ----------
        online_data: numpy.ndarray
            The data used for calculating the monitors which must be of shape
            (m, n) where m is the number of variables (which must match up to
            the number of variables in the training data) and n is the number
            of test samples (which must be at least 2 more than the amount of
            dynamic copies used for training)

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
        if not self.parted:
            raise RuntimeError("Signals have not been partitioned yet")

        # Add lagged copies and perform nonlinear expansion
        online_data = self.process_data(online_data)

        # Standardize and transform data
        centered_data = self.standardization_node.center_similar(online_data)
        features_online = self.transformation_matrix @ centered_data

        # Split slow features into fastest and slowest based on training data
        features_slowest = features_online[:self.Md, :]
        features_fastest = features_online[self.Md:, :]
        # Calculate time derivatives
        features_slowest_derivative = np.diff(features_slowest)/self.delta
        features_fastest_derivative = np.diff(features_fastest)/self.delta

        n = features_slowest.shape[1]

        T_d = np.zeros((n))
        T_e = np.zeros((n))
        S_d = np.zeros((n))
        S_e = np.zeros((n))

        for i in range(n):
            sample_slow = features_slowest[:, i]
            sample_fast = features_fastest[:, i]
            T_d[i] = sample_slow.T @ sample_slow
            T_e[i] = sample_fast.T @ sample_fast

        Omega = self.features_speed
        Omega_slow = Omega[:self.Md]
        Omega_fast = Omega[self.Md:]

        Omega_slow_inv = np.diag(Omega_slow**(-1))
        Omega_fast_inv = np.diag(Omega_fast**(-1))

        for i in range(n - 1):
            sample_slow = features_slowest_derivative[:, i]
            sample_fast = features_fastest_derivative[:, i]
            S_d[i + 1] = sample_slow.T @ Omega_slow_inv @ sample_slow
            S_e[i + 1] = sample_fast.T @ Omega_fast_inv @ sample_fast

        T_d = T_d.reshape((-1, 1))
        T_e = T_e.reshape((-1, 1))
        S_d = S_d.reshape((-1, 1))
        S_e = S_e.reshape((-1, 1))
        stats = np.concatenate((T_d, T_e, S_d, S_e), axis=1)
        return(stats)
