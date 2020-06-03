import numpy as np
import TEP_Import as imp
from SFAClass import SFA
import DataNode


# TODO: Is there a set precision that should be used?
PREC = 6 # Consider any number less than 10^(-PREC) to be 0
MAX_FEATURES = 250

def dataSetup():
    # TODO: Importing data everytime is slow, speed this up somehow
    training_sets = list(imp.importTrainingSets([0]))
    training_set_0 = training_sets[0]
    X = training_set_0[1]
    X = np.delete(X,range(21,42),axis=0)
    return X

def objectSetup(k,order):
    X = dataSetup()
    SlowFeature = SFA(X,k,order)
    SlowFeature.delta = 3
    SlowFeature.train()
    
    return SlowFeature

class DataFeatureProperties:
    order = None
    copies = None
    SFObj = None
    
    def test_dataVariance(self):
        Z = self.SFObj.signals_norm
        num_signals = self.SFObj.m
        num_samples = self.SFObj.N
        cov_matrix = np.matmul(Z,Z.T)/(num_samples-1)
        ident = np.identity(num_signals)
        diff = np.around(cov_matrix - ident,PREC)
        assert not np.all(diff)
        return

    def test_dataMean(self):
        Z = self.SFObj.signals_norm
        Zmeans = Z.mean(axis=1)
        assert not np.all(np.around(Zmeans,PREC))
        return

    def test_sfVariance(self):
        Y = self.SFObj.features
        num_features = self.SFObj.m
        num_samples = self.SFObj.N
        cov_matrix = np.matmul(Y,Y.T)/(num_samples-1)
        ident = np.identity(num_features)
        diff = np.around(cov_matrix - ident,PREC)
        assert not np.all(diff)
        return

    def test_sfOrder(self):
        Y = self.SFObj.features
        num_samples = self.SFObj.N
        num_features = self.SFObj.m
        if num_features > MAX_FEATURES:
            Y = np.copy(Y[:MAX_FEATURES,:])
            num_features = MAX_FEATURES

        precision = 10**(-PREC)
        for i in range(0,num_features):
            for j in range(i+1,num_features):
                prod = np.matmul(Y[i,:],Y[j,:])/(num_samples-1)
                assert abs(prod) < precision
        return
    
class TestDynamicUnexpanded(DataFeatureProperties):
    SFObj = objectSetup(2,1)
            
class TestStaticUnexpanded(DataFeatureProperties):
    SFObj = objectSetup(0,1)

class TestDynamicQuadexpanded(DataFeatureProperties):
    SFObj = objectSetup(2,2)

class TestStaticQuadexpanded(DataFeatureProperties):
    SFObj = objectSetup(0,2)
        
class TestStaticCubeExpanded(DataFeatureProperties):
    SFObj = objectSetup(0,3)
