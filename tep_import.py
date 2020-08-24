""" Functions to import TEP dataset

A set of functions to import the data from the Tennesse Eastman Process
"""

__author__ = "Hussein Saafan"

import numpy as np

_folder_path = "TE_Process/"
_training_sets = []
_test_sets = []

for i in range(22):
    _training_sets.append("d" + str(i).zfill(2) + ".dat")
    _test_sets.append("d" + str(i).zfill(2) + "_te.dat")


def import_training_sets(sets_to_import=range(22)):
    """
    Takes a sequence of integers from 0-21 and returns a zip
    of the TEP training dataset name and data
    """
    if not hasattr(type(sets_to_import), '__iter__'):
        raise TypeError("Expected an iterable object")
    datasetnames = []
    datasets = []
    for val in sets_to_import:
        if not type(val) == int:
            raise TypeError("Expceted an integer value")
        if val > 21 or val < 0:
            raise ValueError("Expected an integer between 0 and 21")
        name = _training_sets[val]
        data = np.loadtxt(_folder_path+name)

        datasetnames.append(name)
        datasets.append(data)
    sets = zip(datasetnames, datasets)
    return(sets)


def import_test_sets(sets_to_import=range(22)):
    """
    Takes a sequence of integers from 0-21 and returns a zip
    of the TEP test dataset name and data
    """
    if not hasattr(type(sets_to_import), '__iter__'):
        raise TypeError("Expected an iterable object")
    datasetnames = []
    datasets = []
    for val in sets_to_import:
        if not type(val) == int:
            raise TypeError("Expceted an integer value")
        if val > 21 or val < 0:
            raise ValueError("Expected an integer between 0 and 21")
        name = _test_sets[val]
        data = np.loadtxt(_folder_path+name)

        datasetnames.append(name)
        datasets.append(data)

    sets = zip(datasetnames, datasets)
    return(sets)


def import_tep_sets():
    """ Imports the sets used for training and testing

    Imports the training set d00 and the test sets for
    IDV(4), IDV(5), and IDV(10)
    """
    training_sets = list(import_training_sets([0]))
    testing_sets = list(import_test_sets([0, 4, 5, 10]))

    X = training_sets[0][1]
    T0 = testing_sets[0][1].T
    T4 = testing_sets[1][1].T
    T5 = testing_sets[2][1].T
    T10 = testing_sets[3][1].T

    ignored_var = list(range(22, 41))
    X = np.delete(X, ignored_var, axis=0)
    T0 = np.delete(T0, ignored_var, axis=0)
    T4 = np.delete(T4, ignored_var, axis=0)
    T5 = np.delete(T5, ignored_var, axis=0)
    T10 = np.delete(T10, ignored_var, axis=0)
    return(X, T0, T4, T5, T10)


if __name__ == "__main__":
    print("This file cannot be run directly, import this module to obtain the",
          "datasets of the Tennessee Eastman process")
