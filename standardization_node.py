""" Tools used for standarizing data

Some classes that are used for data centering and whitening

Classes
-------
Standardization: class
    Centering and whitening using SVD of covariance matrix
IncrementalStandardization: class
    Incremental centering and whitening using CCIPCA [1]_
RecursiveStandardization: class
    Recursive centering and whitening [2]_

References
----------
.. [1] Juyang Weng, Yilu Zhang, & Wey-Shiuan Hwang. (2003). Candid
       covariance-free incremental principal component analysis. IEEE
       Transactions on Pattern Analysis and Machine Intelligence, 25(8),
       1034–1040. <https://doi.org/10.1109/tpami.2003.1217609>
.. [2] Shang, C., Yang, F., Huang, B., & Huang, D. (2018). Recursive Slow
       Feature Analysis for Adaptive Monitoring of Industrial Processes. IEEE
       Transactions on Industrial Electronics (1982), 65(11), 8895–8905.
       <https://doi.org/10.1109/tie.2018.2811358>
"""

__author__ = "Hussein Saafan"

import warnings

import numpy as np
import scipy.linalg as LA
import scipy

from data_node import Node

eps = np.finfo(float).eps


class Standardization(Node):
    """ Batch data centering and whitening

    Attributes
    ----------
    offset: numpy.ndarray
        The average of the signals across all samples
    whitening_matrix: numpy.ndarray
        The whitening transformation matrix used to whiten zero-mean data
    training_data: numpy.ndarray
        The data that was used for getting the offset and whitening matrix
    """
    offset = None
    whitening_matrix = None
    training_data = None

    def __init__(self, data):
        """ Class constructor

        Parameters
        ----------
        data: numpy.ndarray
            The data to use for generating the whitening matrix and offset
            which must be d by n where d is the number of variables and n is
            the number of samples
        """
        self._check_input_data(data)
        self.training_data = data

    def _center(self, data):
        """ Center training data and learn the offset

        Parameters
        ----------
        data: numpy.ndarray
            The data used for learning the offset

        Returns
        -------
        centered_data: numpy.ndarray
            Data with zero mean for all variables
        """
        data_means = data.mean(axis=1).reshape((data.shape[0], 1))
        self.offset = data_means
        centered_data = data - self.offset
        return(centered_data)

    def _whiten(self, data):
        """ Learn the whitening matrix using SVD

        Calculates the covariance matrix and performs SVD to
        learn the whitening matrix (PCA whitening transform)

        Parameters
        ----------
        data: numpy.ndarray
            The data used for learning the whitening matrix

        Returns
        -------
        whitened_data: numpy.ndarray
            Data with identity variance matrix
        """
        Sigma = np.cov(data)
        U, Lambda, UT = np.linalg.svd(Sigma, hermitian=True)

        # Calculate the whitening matrix
        Q = np.diag(Lambda**-(1/2)) @ UT
        self.whitening_matrix = Q
        whitened_data = Q @ data

        return(whitened_data)

    def standardize(self):
        """ Trains the model and returns the standardized data

        Returns
        -------
        standardized_data: numpy.ndarray
            Data with identity variance matrix and zero mean
        """
        centered_data = self._center(self.training_data)
        standardized_data = self._whiten(centered_data)
        return(standardized_data)

    def center_similar(self, data):
        """ Center data using the learned offset

        Parameters
        ----------
        data: numpy.ndarray
            The data to be centered

        Returns
        -------
        centered_data: numpy.ndarray
            Data minus the offset
        """
        centered_data = data - self.offset
        return(centered_data)

    def whiten_similar(self, data):
        """ Whiten data using learned whitening matrix

        Parameters
        ----------
        data: numpy.ndarray
            The data to be whitened

        Returns
        -------
        whitened_data: numpy.ndarray
        """
        whitened_data = self.whitening_matrix @ data
        return(whitened_data)

    def standardize_similar(self, data):
        """ Standardized data using learned model

        Parameters
        ----------
        data: numpy.ndarray
            The data to be standardized

        Returns
        -------
        standardized_data: numpy.ndarray
            Data that has been centered and whitened
        """
        self._check_input_data(data)
        centered_data = self.center_similar(data)
        standardized_data = self.whiten_similar(centered_data)
        return(standardized_data)


class IncrementalStandardization(Node):
    """ Incremental data centering and whitening

    Attributes
    ----------
    offset: numpy.ndarray
        The average of the signals across all samples
    whitening_matrix: numpy.ndarray
        The whitening transformation matrix used to whiten zero-mean data
    eigensystem: numpy.ndarray
        Matrix of eigenvectors multiplied by their respective eigenvalues
    num_components: int
        The number of principal components to keep
    _count: int
        The number of samples that have been passed to the model so far
    """
    offset = None
    whitening_matrix = None
    eigensystem = None
    num_components = None
    _count = None

    def __init__(self, first_sample, num_components):
        """ Class constructor

        Initializes object based on first sample passed through

        Parameters
        ----------
        first_sample: numpy.ndarray
            The first sample to initialize the object based off of which must
            be of shape (d, 1) where d is the number of variables
        num_components: int
            The number of principal components to keep for whitening. The more
            components kept, the longer it will take for convergance.
        """
        self._check_input_data(first_sample)
        self.offset = np.zeros_like(first_sample)
        self._store_num_components(num_components)

        # Initialize eigensystem
        d = first_sample.shape[0]
        self.eigensystem = np.ones((d, self.num_components))
        self._count = 0

    def _store_num_components(self, num_components):
        """ Stores the the number of principal components

        Parameters
        ----------
        num_components: int
            Number of principal components to keep for whitening
        """
        if not isinstance(num_components, int):
            raise TypeError(f"{type(m)} is not a valid input for "
                            "num_components , expected int")
        elif num_components < 1:
            raise ValueError("Expected a positive integer input for "
                             "number of principal components")
        else:
            self.num_components = num_components
        return

    def _center(self, sample, eta):
        """ Center a sample and update the offset

        Parameters
        ----------
        sample: numpy.ndarray
            The sample used for learning the offset
        eta: float
            The learning rate

        Returns
        -------
        centered_sample: numpy.ndarray
            Sample minus the learned offset
        """
        self.offset = (1 - eta) * self.offset + eta * sample
        centered_sample = sample - self.offset
        return(centered_sample)

    def _CCIPA(self, eigensystem, sample, eta):
        """ Candid Covariance Free Incremental Principal Component Analysis

        Perform CCIPCA [1]_ and update the eigensystem estimate

        Parameters
        ----------
        eigensystem: numpy.ndarray
            An estimate of the eigensystem where each column of the matrix is
            an eigenvector multiplied by its respective eigenvalue
        sample: numpy.ndarray
            The sample used to update the eigensystem
        eta: float
            The learning rate

        Returns
        -------
        eigensystem: numpy.ndarray
            The updated eigensystem estimate where each column of the matrix is
            an eigenvector multiplied by its respective eigenvalue

        References
        ----------
        .. [1] Juyang Weng, Yilu Zhang, & Wey-Shiuan Hwang. (2003). Candid
        covariance-free incremental principal component analysis. IEEE
        Transactions on Pattern Analysis and Machine Intelligence, 25(8),
        1034–1040. <https://doi.org/10.1109/tpami.2003.1217609>
        """
        max_iter = int(np.min([self.num_components, self._count + 1]))
        u = sample.reshape((-1, 1))
        u_orig = np.copy(u)
        resid_sum = np.zeros_like(u)
        for i in range(max_iter):
            u = u_orig - resid_sum
            if i == self._count:
                eigensystem[:, i] = u.flat
            else:
                # Get previous estimate of v
                v_prev = eigensystem[:, i].reshape((-1, 1))
                v_prev_norm = LA.norm(v_prev) + eps

                # Get new estimate of v
                historical = (1 - eta) * v_prev
                projection = (u_orig.T @ v_prev) / v_prev_norm
                update = eta * u * projection
                v_new = historical + update

                # Store new estimate
                eigensystem[:, i] = v_new.flat

                # Update sum from projections
                v_norm = LA.norm(v_new) + eps
                resid_sum += ((u_orig.T @ v_new) * v_new)/(v_norm ** 2)
        return(eigensystem)

    def _get_whitening_matrix(self, eigensystem, zca_whitening=False):
        """ Calculate the whitening matrix

        Parameters
        ----------
        eigensystem: numpy.ndarray
            The current eigensystem estimate
        zca_whitening: boolean (default: False)
            Use ZCA whitening matrix instead of PCA

        Returns
        -------
        whitening_matrix: numpy.ndarray
            The whitening matrix estimate
        """
        D = np.zeros((self.num_components))
        U = np.zeros_like(eigensystem)
        # Calculate the eigenvectors and the diagonal matrix
        for i in range(self.num_components):
            eigenvalue = LA.norm(eigensystem[:, i]) + eps
            U[:, i] = eigensystem[:, i] / eigenvalue
            D[i] = eigenvalue ** (-1/2)

        # Calculate the whitening matrix
        S = U @ np.diag(D)
        if zca_whitening:
            S = S @ U.T
        whitening_matrix = S.T

        return(whitening_matrix)

    def _whiten(self, sample, eta):
        """ Update the model and whiten the current sample

        Parameters
        ----------
        sample: numpy.ndarray
            The sample used for updating the model
        eta: float
            The learning rate

        Returns
        -------
        whitened_sample: numpy.ndarray
            Sample that has been transformed by the current whitening matrix
            estimate
        """
        sample = sample.reshape((-1, 1))
        self.eigensystem = self._CCIPA(self.eigensystem, sample, eta)
        self.whitening_matrix = self._get_whitening_matrix(self.eigensystem)
        whitened_sample = self.whitening_matrix @ sample
        return(whitened_sample)

    def standardize_online(self, sample, eta=0.01):
        """ Updates the model and returns the standardized sample

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to update the model with
        eta: float
            The learning rate

        Returns
        -------
        standardized_sample: numpy.ndarray
            Sample that has been centered and transformed by the current
            whitening matrix estimate
        """
        self._check_input_data(sample)
        centered_sample = self._center(sample, eta)
        standardized_sample = self._whiten(centered_sample, eta)
        self._count += 1
        return(standardized_sample)

    def update_CCIPA(self, sample, eta):
        """ Updates and returns the model without transforming data

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to update the model with
        eta: float
            The learning rate

        Returns
        -------
        eigensystem: numpy.ndarray
            The current eigensystem estimate
        """
        centered_data = self._center(sample, eta)
        self.eigensystem = self._CCIPA(self.eigensystem, centered_data, eta)
        self._count += 1
        return(self.eigensystem)

    def center_similar(self, sample):
        """ Center sample using the learned offset

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to be centered

        Returns
        -------
        centered_sample: numpy.ndarray
            Sample minus the offset
        """
        centered_sample = sample - self.offset
        return(centered_sample)

    def whiten_similar(self, sample):
        """ Whiten sample using learned whitening transform

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to be whitened

        Returns
        -------
        whitened_sample: numpy.ndarray
        """
        whitened_sample = self.whitening_matrix @ sample
        return(whitened_sample)

    def standardize_similar(self, sample):
        """ standardized sample using learned model

        The model isn't updated when this method is used

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to be whitened

        Returns
        -------
        standardized_sample: numpy.ndarray
            Sample that has been centered and whitened
        """
        self._check_input_data(sample)
        centered_sample = self.center_similar(sample)
        standardized_sample = self.whiten_similar(centered_sample)
        return(standardized_sample)


class RecursiveStandardization(Node):
    """ Recursive data centering and whitening

    Attributes
    ----------
    offset: numpy.ndarray
        The average of the signals across all samples
    offset_delta: numpy.ndarray
        The derivative of the average
    covariance: numpy.ndarray
        The estimated covariance matrix of the input data
    whitening_matrix: numpy.ndarray
        The whitening transformation matrix used to whiten zero-mean data

    References
    ----------
    .. [1] Li, W., Yue, H., Valle-Cervantes, S., & Qin, S. (2000). Recursive
           PCA for adaptive process monitoring. Journal of Process Control, 1
           (5), 471–486. <https://doi.org/10.1016/s0959-1524(00)00022-6>
    """
    offset = None
    offset_delta = None
    covariance = None
    whitening_matrix = None

    def __init__(self, first_sample):
        """ Class constructor

        Initializes object based on first sample passed through

        Parameters
        ----------
        first_sample: numpy.ndarray
            The first sample to initialize the object based off of which must
            be of shape (d, 1) where d is the number of variables
        """
        self._check_input_data(first_sample)
        self.offset = np.zeros_like(first_sample)
        self.offset_delta = np.zeros_like(first_sample)
        d = first_sample.shape[0]
        self.covariance = np.zeros((d, d))

    def _center(self, sample, eta):
        """ Center a sample and update the offset and its derivative

        Parameters
        ----------
        sample: numpy.ndarray
            The sample used for learning the offset
        eta: float
            The learning rate

        Returns
        -------
        centered_sample: numpy.ndarray
            Sample minus the learned offset
        """
        prev_offset = np.copy(self.offset)
        self.offset = (1 - eta) * self.offset + eta * sample
        self.offset_delta = self.offset - prev_offset
        centered_sample = sample - self.offset

        return(centered_sample)

    def _update_sample_cov(self, sample, eta):
        """ Update the covariance matrix estimate

        Parameters
        ----------
        sample: numpy.ndarray
            The sample used for updating the matrix
        eta: float
            The learning rate
        """
        '''
        prev_cov = self.covariance
        recenter = self.offset_delta @ self.offset_delta.T
        new_cov = eta * (prev_cov + recenter) + (1 - eta) * (sample @ sample.T)
        self.covariance = new_cov
        '''
        self.covariance = (1 - eta) * self.covariance + eta * sample @ sample.T
        return

    def _whiten(self, sample, eta):
        """ Update the model and whiten the current sample

        Parameters
        ----------
        sample: numpy.ndarray
            The sample used for updating the model
        eta: float
            The learning rate

        Returns
        -------
        whitened_sample: numpy.ndarray
            Sample that has been transformed by the current whitening matrix
            estimate
        """
        # TODO: Replace SVD with rank one modification as in paper
        U, L, UT = LA.svd(self.covariance)
        self.whitening_matrix = U @ np.diag(L ** (-1/2))
        self.whitening_eigenvals = L
        whitened_sample = self.whitening_matrix.T @ sample
        return(whitened_sample)

    def standardize_online(self, sample, eta=0.01):
        """ Updates the model and returns the standardized sample

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to update the model with
        eta: float
            The learning rate

        Returns
        -------
        standardized_sample: numpy.ndarray
            Sample that has been centered and transformed by the current
            whitening matrix estimate
        """
        self._check_input_data(sample)
        centered_sample = self._center(sample, eta)
        self._update_sample_cov(centered_sample, eta)
        standardized_sample = self._whiten(centered_sample, eta)
        return(standardized_sample)

    def center_similar(self, sample):
        """ Center sample using the learned offset

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to be centered

        Returns
        -------
        centered_sample: numpy.ndarray
            Sample minus the offset
        """
        centered_sample = sample - self.offset
        return(centered_sample)

    def whiten_similar(self, sample):
        """ Whiten sample using learned whitening transform

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to be whitened

        Returns
        -------
        whitened_sample: numpy.ndarray
        """
        whitened_sample = self.whitening_matrix @ sample
        return(whitened_sample)

    def standardize_similar(self, sample):
        """ standardized sample using learned model

        The model isn't updated when this method is used

        Parameters
        ----------
        sample: numpy.ndarray
            The sample to be whitened

        Returns
        -------
        standardized_sample: numpy.ndarray
            Sample that has been centered and whitened
        """
        self._check_input_data(sample)
        centered_sample = self.center_similar(sample)
        standardized_sample = self.whiten_similar(centered_sample)
        return(standardized_sample)
