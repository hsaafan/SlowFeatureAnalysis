""" PyTest tests of sfa.py"""
__author__ = "Hussein Saafan"

import numpy as np

import tep_import as imp
from sfa import SFA

_prec = 6  # Consider any number less than 10^(-_prec) to be 0
_max_features = 250


def dataSetup():
    training_sets = list(imp.importTrainingSets([0]))
    training_set_0 = training_sets[0]
    X = training_set_0[1]
    X = np.delete(X, range(21, 42), axis=0)
    return X


def objectSetup(k, order):
    X = dataSetup()
    SlowFeature = SFA(X, k, order)
    SlowFeature.delta = 3
    SlowFeature.train()

    return SlowFeature


class DataFeatureProperties:
    order = None
    copies = None
    SFObj = None

    def test_dataVariance(self):
        Z = self.SFObj.standardized_data
        num_signals = self.SFObj.standardized_data.shape[0]
        num_samples = self.SFObj.standardized_data.shape[1]
        cov_matrix = np.cov(Z)
        ident = np.identity(num_signals)
        diff = np.around(cov_matrix - ident, _prec)
        assert not np.all(diff)
        return

    def test_dataMean(self):
        Z = self.SFObj.standardized_data
        Zmeans = Z.mean(axis=1)
        assert not np.all(np.around(Zmeans, _prec))
        return

    def test_sfVariance(self):
        Y = self.SFObj.features
        num_features = self.SFObj.features.shape[0]
        num_samples = self.SFObj.features.shape[1]
        cov_matrix = np.cov(Y)
        ident = np.identity(num_features)
        diff = np.around(cov_matrix - ident, _prec)
        assert not np.all(diff)
        return

    def test_sfOrder(self):
        Y = self.SFObj.features
        num_features = self.SFObj.features.shape[0]
        num_samples = self.SFObj.features.shape[1]
        if num_features > _max_features:
            Y = np.copy(Y[:_max_features, :])
            num_features = _max_features

        precision = 10**(-_prec)
        for i in range(0, num_features):
            for j in range(i+1, num_features):
                prod = (Y[i, :] @ Y[j, :])/(num_samples-1)
                assert abs(prod) < precision
        return


class TestDynamicUnexpanded(DataFeatureProperties):
    SFObj = objectSetup(2, 1)


class TestStaticUnexpanded(DataFeatureProperties):
    SFObj = objectSetup(0, 1)


class TestDynamicQuadexpanded(DataFeatureProperties):
    SFObj = objectSetup(2, 2)


class TestStaticQuadexpanded(DataFeatureProperties):
    SFObj = objectSetup(0, 2)


class TestStaticCubeExpanded(DataFeatureProperties):
    SFObj = objectSetup(0, 3)
