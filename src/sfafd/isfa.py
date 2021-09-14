""" Iterative Slow Feature Analysis

Implementation of Iterative Slow Feature Analysis [1]_. This module isn't
ready for implementation/testing.

Classes
-------
ISFA: class
    The class used for Iterative Slow Feature Analysis

References
----------
.. [1] Liangpei Zhang, Chen Wu, & Bo Du. (2014). Automatic Radiometric
       Normalization for Multitemporal Remote Sensing Imagery With Iterative
       Slow Feature Analysis. IEEE Transactions on Geoscience and Remote
       Sensing, 52(10), 6141–6155. <https://doi.org/10.1109/TGRS.2013.2295263>
"""

__author__ = "Hussein Saafan"

import warnings

import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
import scipy.stats

import tepimport as imp


class ISFA:
    """
    Iterative slow feature analysis class that takes a set of input signals
    and calculates the slow features for the inputs
    """
    num_vars = 0                  # Number of variables in a sample
    num_samples = 0               # Number of samples per "image"
    weights = None                # How samples are weighted in an "image"
    weights_total = None          # Sum of weights

    offset = None                 # Offset for each variable
    stdv = None                   # Standard deviation for each variable
    Lambda = None                 # Current eigenvalues
    tol = 0.05                    # Stopping criteria for eigenvalues

    def __init__(self, num_vars, num_samples, weights=None):
        """
        Constructor method for ISFA object that takes expected dimensions of
        inputs and weights
        Input "images" are matrices of different samples
        Each "pixel" is a sample of the "image" that contains the same
        number of variables
        Each variable here is akin to a "spectral band"

        num_vars and num_samples takes an integer value (>1)
        weights takes an ndarray of shape (num_samples)
        """
        self._store_num_vars(num_vars)
        self._store_num_samples(num_samples)
        self._store_weights(weights)

        # Initial offset and stdv
        self.offset = np.zeros((num_vars, 1))
        self.stdv = np.ones((num_vars, 1))

        return

    def _store_num_vars(self, num_vars):
        """
        Perform checks on number of variables and stores the number
        """
        if type(num_vars) == int:
            if num_vars > 1:
                self.num_vars = num_vars
            else:
                raise ValueError("Number of variables must be at least 2")
        else:
            raise TypeError("Expected an integer number of variables")
        return

    def _store_num_samples(self, num_samples):
        """
        Perform checks on number of samples and stores the number
        """
        if type(num_samples) == int:
            if num_samples > 1:
                self.num_samples = num_samples
            else:
                raise ValueError("Number of samples must be at least 2")
        else:
            raise TypeError("Expected an integer number of samples")
        return

    def _store_weights(self, weights):
        """
        Perform checks on weights and store array
        """
        if not (weights is None):
            if isinstance(weights, np.ndarray):
                if weights.ndim == 1:
                    if weights.size == self.num_samples:
                        self.weights = weights
                        self.weights_total = np.sum(weights)
                    else:
                        raise ValueError("Each sample must have one weight")
                else:
                    raise TypeError("Weights must be a 1D ndarray object")
            else:
                raise TypeError("Weights must be a 1D ndarray object")
        else:
            # If not weights are passed, assign unit weights
            self.weights = np.ones(self.num_samples)
            self.weights_total = self.num_samples
        return

    def _normalize_data(self, data):
        """
        Takes a new set of weights and recalculates normalized data
        """
        centered_data = self._center(data)
        normalized_data = self._whiten(centered_data)
        return(normalized_data)

    def _weight_data(self, data):
        weighted_data = np.multiply(self.weights, data)
        return(weighted_data)

    def _center(self, data):
        """
        Centers all variables around 0
        """
        weighted_data = self._weight_data(data)
        self.offset = (np.sum(weighted_data, axis=1)
                       / self.weights_total).reshape(self.num_vars, 1)
        centered_data = data - self.offset
        return(centered_data)

    def _whiten(self, centered_data):
        """
        Scale variables to unit variance if they have been centered
        """
        sqr = np.multiply(centered_data, centered_data)
        weighted_sqr = self._weight_data(sqr)
        stdv = (np.sum((weighted_sqr/self.weights_total), axis=1)
                ** (1/2)).reshape(self.num_vars, 1)
        whitened_data = centered_data / stdv
        return(whitened_data)

    def _calculate_AB(self, x_hat, y_hat):
        A = np.zeros((self.num_vars, self.num_vars))
        B = np.zeros((self.num_vars, self.num_vars))
        for i in range(self.num_samples):
            x_sample = x_hat[:, i].reshape((self.num_vars, 1))
            y_sample = y_hat[:, i].reshape((self.num_vars, 1))

            diff = x_sample - y_sample
            xsqr = x_sample @ x_sample.T
            ysqr = y_sample @ y_sample.T

            grp_A = diff @ diff.T
            grp_B = xsqr + ysqr
            A += grp_A
            B += grp_B

        A /= self.weights_total
        B /= 2*self.weights_total
        a, b, c = np.svd(B)
        # A = np.cov(x_hat - y_hat)
        # B = (np.cov(x_hat) + np.cov(y_hat))/(2*self.weights_total)
        return(A, B)

    def _calculate_trans_mat(self, A, B):
        """
        Solving the generalized eigenvalue problem
        """
        # TODO: Make this more efficient
        eps = 1e-5
        ident = np.eye(B.shape[0])
        B_inv = LA.inv(B + eps * ident)
        C = B_inv @ A

        Lambda, W = LA.eigh(C)
        Lambda, W = self._order(Lambda, W)
        for col in range(W.shape[1]):
            wj = W[:, col]
            W[:, col] = wj / ((wj.T @ B @ wj)**(1/2))
            for i in range(col):
                temp = wj.T @ B @ W[:, i]
                temp2 = wj.T @ B @ wj
            pass
        return(Lambda, W)

    def _calculate_slow_features(self, W, x_hat, y_hat):
        """
        Calculates slow features given a transformation matrix and
        normalized data
        """
        SFA = np.empty((self.num_vars, self.num_samples))
        for j in range(self.num_vars):
            w_vec = W[:, j]
            SFA[j, :] = w_vec.T @ x_hat - w_vec.T @ y_hat
        return(SFA)

    def _calculate_weights(self, SFA, Lambda):
        """
        Calculates the weights of the samples as the probability that
        T is less than chi squared with num_vars degrees of freedom
        """
        T = np.zeros((self.num_samples))
        weights = np.zeros((self.num_samples))
        for i in range(self.num_samples):
            summation = 0
            for j in range(self.num_vars):
                summation += (SFA[j, i] ** 2)/(Lambda[j])
            T[i] = summation
            weights[i] = 1 - scipy.stats.chi2.cdf(T[i], self.num_vars)
        return(weights)

    def _iterate(self, x, y, Lambda_prev):
        """
        Iterates through algorithm until the eigenvalues converge
        Weights are updated after each iteration
        """
        finished = False
        err = 1e20

        while not finished:
            # Normalize data
            x_hat = self._normalize_data(x)
            y_hat = self._normalize_data(y)

            # Calculate transformation matrix and speeds
            A, B = self._calculate_AB(x_hat, y_hat)
            Lambda, W = self._calculate_trans_mat(A, B)

            # Checks if speeds have converged
            prev_err = err
            err = LA.norm(Lambda - Lambda_prev)
            rel_err = np.abs(err - prev_err)/np.abs(prev_err)
            if rel_err < self.tol:
                finished = True
            else:
                Lambda_prev = np.copy(Lambda)
            # Calculates slow features and updates weights
            SFA = self._calculate_slow_features(W, x_hat, y_hat)
            self._store_weights(self._calculate_weights(SFA, Lambda))
        return(SFA, W, Lambda)

    def _order(self, eigenvalues, eigenvectors):
        """
        Orders a set of eigenvalue/eigenvector pairs by ascending eigenvalues
        """
        p = eigenvalues.argsort()
        vals = eigenvalues[p]
        vecs = eigenvectors[p, :]

        return(vals, vecs)

    def train(self, x, y):
        """
        Takes in 2 matrices of shape (num_vars,num_samples)
        Outputs the slow features
        """
        if not(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            raise TypeError("Data passed must be a numpy ndarray")
        elif not(x.shape == (self.num_vars, self.num_samples) and
                 x.shape == (self.num_vars, self.num_samples)):
            warnings.warn("Data passed not the right shape", RuntimeWarning)
            SFA = np.zeros((self.num_vars, self.num_samples))
        else:
            if self.Lambda is None:
                # First training data passed
                self.Lambda = np.ones((self.num_vars))
            SFA, W, self.Lambda = self._iterate(x, y, self.Lambda)

        return(SFA)


if __name__ == "__main__":
    # Each image is a set of 5 samples
    # Each sample has a set number of variables (or band values) in it
    # So each sample is a band and each value is a band value

    # Import TEP and set up input data
    X, T4, T5, T10 = imp.import_tep_sets(lagged_samples=0)

    num_samples = 100
    num_vars = X.shape[0]
    total_samples = X.shape[1]
    SlowFeature = ISFA(num_vars, num_samples)
    Y = np.ones_like(X)
    W = np.ones((X.shape[1]))

    for i in range(num_samples, total_samples, num_samples):
        Y[:, i-num_samples:i] = SlowFeature.train(X[:, i-num_samples:i],
                                                  X[:, i:i+num_samples])
        W[i-num_samples:i] = SlowFeature.weights
    """
    for i in range(num_samples, total_samples):
        Y[:,i] = SlowFeature.train(X[:,i:i+num_samples],
                                   X[:,i-num_samples:i])[:,0]
    """
    W = W[:num_samples]
    Y = Y[:, :-num_samples]
    mid = int(Y.shape[0]/2)
    slowest = Y[:5, :]
    middle = Y[mid:mid+5, :]
    fastest = Y[-5:, :]
    plt.figure("Slow Features")
    for i in range(5):
        plt.subplot(5, 3, 3*i+1)
        plt.plot(slowest[i, :])
        if i == 0:
            plt.title("Slowest")

        plt.subplot(5, 3, 3*i+2)
        plt.plot(middle[i, :])
        if i == 0:
            plt.title("Middle")

        plt.subplot(5, 3, 3*i+3)
        plt.plot(fastest[i, :])
        if i == 0:
            plt.title("Fastest")

    plt.figure("all")
    for i in range(num_vars):
        plt.subplot(11, 3, i+1)
        plt.plot(Y[i, :])
    plt.show()
