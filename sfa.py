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
import scipy.stats as stats

from data_node import Node
from standardization_node import Standardization


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
    delta: float
        The time between samples in the data, default is 1
    training_samples: int
        The number of samples the model was trained on
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
    standardization_node: Standardization
        Used to standardize training and online data
    training_data: numpy.ndarray
        The data used to train the model
    standardized_data: numpy.ndarray
        The data used to train the model that has been standardized
    features_speed: numpy.ndarray
        The singular values of the transformation matrix that were found using
        singular value decompisition. These values are proportional to the
        speeds of the features.
    Md: int
        The amount of features that are considered "slow"
    Me: int
        The amount of features that are considered "fast"
    """
    trained = False
    parted = False
    delta = 1
    training_samples = None

    features = None

    transformation_matrix = None
    standardization_node = None
    training_data = None
    standardized_data = None

    features_speed = None
    Md = None
    Me = None

    @property
    def slow_features(self):
        if not self.parted:
            raise RuntimeError("Signals have not been partitioned yet")
        return(self.features[:self.Me, :])

    @property
    def fast_features(self):
        if not self.parted:
            raise RuntimeError("Signals have not been partitioned yet")
        return(self.features[self.Me:, :])

    def __init__(self, data, dynamic_copies=None, expansion_order=None):
        """ Class constructor

        Parameters
        ----------
        data: numpy.ndarray
            The data to use for training the model which must be d by n where
            d is the number of variables and n is the number of samples
        dynamic_copies: int
            The amount of lagged copies to add to the data
        expansion_order: int
            The order of nonlinear expansion to perform on the data
        """
        self._check_input_data(data)
        d = data.shape[0]
        n = data.shape[1]
        super().__init__(d, dynamic_copies, expansion_order)
        self.training_samples = n
        self.training_data = self.process_data(data)

    def _standardize_training(self):
        """ Updates the standardization model using the training data """
        self.standardization_node = Standardization(self.training_data)
        self.standardized_data = self.standardization_node.standarize()
        return

    def _transform_training(self):
        """ Transforms whitened data into features and updates the model """
        # Approximate the first order time derivative of the signals
        Zdot = np.diff(self.standardized_data)/self.delta
        # SVD of the var-cov matrix of Zdot
        Zdotcov = np.cov(Zdot)
        PT, Omega, P = np.linalg.svd(Zdotcov, hermitian=True)
        # The loadings are ordered to find the slowest varying features
        self.features_speed, P = self._order(Omega, P)
        # Calculate transformation matrix and features
        S = self.standardization_node.whitening_matrix
        self.transformation_matrix = P @ S
        self.features = P @ self.standardized_data
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
        p = eigenvalues.argsort()
        eigenvalues = eigenvalues[p]
        eigenvectors = eigenvectors[p, :]
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

        data = self.standardization_node.standarize_similar(self.training_data)
        # Find slowness of input signals
        xdot = np.diff(data)/self.delta
        mdot = xdot.shape[0]
        Ndot = xdot.shape[1]
        signal_speed = np.zeros(mdot)
        for i in range(mdot):
            signal_derivative = xdot[i, :]
            signal_speed[i] = np.dot(signal_derivative, signal_derivative)/Ndot

        # Find where to partition slow features
        threshold = np.quantile(signal_speed, 1-q, interpolation='lower')
        for i in range(self.features_speed.size):
            if self.features_speed[i] > threshold:
                self.Me = i
                self.Md = self.features_speed.size - i
                break
        self.parted = True
        self.calculate_crit_values()
        return

    def partition_manual(self, Me):
        """ Manually select where to make the cut from slow to fast features

        Parameters
        ----------
        Me: int
            The index of the slow feature where the cuttoff happens. Anything
            less than Me is considered slow and the remainder are considered
            fast.
        """
        self.Me = Me
        self.Md = self.features_speed.size - Me
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
        N = self.training_samples
        Md = self.Md
        Me = self.Me
        gd = (Md*(N**2-2*N))/((N-1)*(N-Md-1))
        ge = (Me*(N**2-2*N))/((N-1)*(N-Me-1))

        T_d_crit = stats.chi2.ppf(p, Md)
        T_e_crit = stats.chi2.ppf(p, Me)
        S_d_crit = gd*stats.f.ppf(p, Md, N-Md-1)
        S_e_crit = ge*stats.f.ppf(p, Me, N-Me-1)

        return(T_d_crit, T_e_crit, S_d_crit, S_e_crit)

    def calculate_monitors(self, online_data):
        """ Calculate monitoring statistics for test data

        Parameters
        ----------
        online_data: numpy.ndarray
            The data used for calculating the monitors which must be of shape
            (d, n) where d is the number of variables (which must match up to
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
        features_slowest = features_online[:self.Me, :]
        features_fastest = features_online[self.Me:, :]
        # Calculate time derivatives
        features_slowest_derivative = np.diff(features_slowest)/self.delta
        features_fastest_derivative = np.diff(features_fastest)/self.delta

        N = features_slowest.shape[1]
        N_dot = features_slowest_derivative.shape[1]

        T_d = np.zeros((N))
        T_e = np.zeros((N))
        S_d = np.zeros((N_dot))
        S_e = np.zeros((N_dot))

        for i in range(N):
            sample_slow = features_slowest[:, i]
            sample_fast = features_fastest[:, i]
            T_d[i] = sample_slow.T @ sample_slow
            T_e[i] = sample_fast.T @ sample_fast

        Omega = self.features_speed
        Omega_slow = Omega[:self.Me]
        Omega_fast = Omega[self.Me:]

        Omega_slow_inv = np.diag(Omega_slow**(-1))
        Omega_fast_inv = np.diag(Omega_fast**(-1))

        for i in range(N_dot):
            sample_slow = features_slowest_derivative[:, i]
            sample_fast = features_fastest_derivative[:, i]
            S_d[i] = sample_slow.T @ Omega_slow_inv @ sample_slow
            S_e[i] = sample_fast.T @ Omega_fast_inv @ sample_fast

        return(T_d, T_e, S_d, S_e)
