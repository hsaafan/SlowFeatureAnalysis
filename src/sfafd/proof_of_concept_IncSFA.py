""" Proof of Concept of Incremental Slow Feature Analysis

A recreation of the proof of concept given in the following paper:

Kompella, Varun Raj, Matthew Luciw, and Jürgen Schmidhuber. “Incremental Slow
Feature Analysis: Adaptive Low-Complexity Slow Feature Updating from
High-Dimensional Input Streams.” Neural computation 24.11 (2012): 2994–3024.
Web.
"""
__author__ = "Hussein Saafan"

import numpy as np
import matplotlib.pyplot as plt

from incsfa import IncSFA


def data_poc(samples):
    t = np.linspace(0, 4*np.pi, num=samples)
    X = np.empty((2, samples))
    X[0, :] = np.sin(t) + np.power(np.cos(11*t), 2)
    X[1, :] = np.cos(11*t)
    return(X)


def RMSE(epoch_data, true_data):
    J = epoch_data.shape[0]
    T = epoch_data.shape[1]
    RMSE = np.zeros(J)
    for j in range(J):
        RMSE[j] = np.power(np.sum(np.power(epoch_data[j, :] - true_data[j, :],
                                           2))/T, 0.5)
    return(RMSE)


if __name__ == "__main__":
    # Proof of concept from paper
    X = data_poc(2000)
    theta = [20, 200, 4, 5000, 0.08, 0.08, -1]
    num_vars = 2
    J = 5
    K = 5
    n = 2
    epochs = 20

    SlowFeature = IncSFA(num_vars, J, K, theta, n)
    Y = np.zeros((J, X.shape[1]*epochs))

    for j in range(epochs):
        for i in range(X.shape[1]):
            run = SlowFeature.add_data(X[:, i])
            Y[:, j*X.shape[1]+i] = (run[0]).reshape((J))

    plt.subplot(2, 1, 1)
    plt.plot(X[0, :])
    plt.subplot(2, 1, 2)
    plt.plot(X[1, :])
    plt.figure()
    for i in range(J):
        plt.subplot(J, 1, i+1)
        plt.plot(Y[i, -2000:])
    plt.show()

    """
    data = np.copy(X)
    for j in range(1, epochs):
        if j == 59:
            X_2 = np.empty_like(X)
            x1 = np.copy(X[0, :])
            x2 = np.copy(X[1, :])
            X_2[0, :] = x2
            X_2[1, :] = x1
            data_2 = np.copy(X_2)
            for k in range(60, epochs):
                data_2 = np.concatenate((data_2, X_2), axis=1)
            SlowFeature_switch = SFA(data_2, None, n)
            true_data_2 = SlowFeature_switch.train()
            break
        data = np.concatenate((data, X), axis=1)

    SlowFeature_orig = SFA(data, expansion_order=n)

    if epochs > 59:
        true_data = np.concatenate((SlowFeature_orig.train(), true_data_2),
                                   axis=1)
    else:
        true_data = SlowFeature_orig.train()

    SlowFeature = IncSFA(num_vars, J, K, theta, n)
    Y = np.zeros((J, X.shape[1]*epochs))
    Z = np.zeros((K, X.shape[1]*epochs))
    err = np.zeros((J, epochs))

    for j in range(epochs):
        if j == 59:
            x1 = np.copy(X[0, :])
            x2 = np.copy(X[1, :])
            X[0, :] = x2
            X[1, :] = x1

        for i in range(X.shape[1]):
            run = SlowFeature.add_data(X[:, i])
            Y[:, j*X.shape[1]+i] = (run[0]).reshape((J))
            if SlowFeature.z_curr is not None:
                Z[:, X.shape[1]*j+i] = SlowFeature.z_curr.reshape(K)

        epoch_data = Y[:, j*X.shape[1]:(j+1)*X.shape[1]]
        epoch_true = true_data[:, j*X.shape[1]:(j+1)*X.shape[1]]
        err[:, j] = RMSE(epoch_data, epoch_true)
    """
