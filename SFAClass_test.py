import numpy as np
import TEP_Import as imp
from SFAClass import SFA

# TODO: Is there a set precision that should be used?
PREC = 6 # Consider any number less than 10^(-PREC) to be 0


def dataSetup():
    # TODO: Importing data everytime is slow, speed this up somehow
    training_sets = list(imp.importTrainingSets([0]))
    training_set_0 = training_sets[0]
    X = training_set_0[1]
    
    return X

def objectSetup(k,order):
    X = dataSetup()
    SlowFeature = SFA(X)
    SlowFeature.train(k,order)
    return SlowFeature


def dataVariance(k,order):
    SlowFeature = objectSetup(k,order)
    Z = SlowFeature.signals_norm
    cov_matrix = np.matmul(Z,Z.T)

    num_features = Z.shape[0]
    ident = np.identity(num_features)

    diff = np.around(cov_matrix - ident,PREC)
    
    assert not np.all(diff)
    return

def dataMean(k,order):
    SlowFeature = objectSetup(k,order)
    Z = SlowFeature.signals_norm
    Zmeans = Z.mean(axis=1)

    assert not np.all(np.around(Zmeans,PREC))
    return

def sfVariance(k,order):
    SlowFeature = objectSetup(k,order)
    Y = SlowFeature.slow_features
    cov_matrix = np.matmul(Y,Y.T)

    num_features = Y.shape[0]
    ident = np.identity(num_features)

    diff = np.around(cov_matrix - ident,PREC)
    
    assert not np.all(diff)
    return

def sfOrder(k,order):
    SlowFeature = objectSetup(k,order)
    Y = SlowFeature.slow_features
    
    Ydot = Y[:,1:] - Y[:,:-1]
    num_features = Y.shape[0]

    precision = 10**(-PREC)
    for i in range(0,num_features):
        for j in range(i+1,num_features):
            prod = np.matmul(Y[i,:],Y[j,:])
            assert abs(prod) < precision
    return

def test_dataVariance_dynamic_quadexpanded():
    dataVariance(4,2)
    return

def test_dataMean_dynamic_quadexpanded():
    dataMean(4,2)
    return

def test_sfVariance_dynamic_quadexpanded():
    sfVariance(4,2)
    return

def test_sfOrder_dynamic_quadexpanded():
    sfOrder(4,2)
    return

def test_dataVariance_dynamic_unexpanded():
    dataVariance(4,1)
    return

def test_dataMean_dynamic_unexpanded():
    dataMean(4,1)
    return

def test_sfVariance_dynamic_unexpanded():
    sfVariance(4,1)
    return

def test_sfOrder_dynamic_unexpanded():
    sfOrder(4,1)
    return

def test_dataVariance_static_unexpanded():
    dataVariance(0,1)
    return

def test_dataMean_static_unexpanded():
    dataMean(0,1)
    return

def test_sfVariance_static_unexpanded():
    sfVariance(0,1)
    return

def test_sfOrder_static_unexpanded():
    sfOrder(0,1)
    return

def test_dataVariance_static_quadexpanded():
    dataVariance(0,2)
    return

def test_dataMean_static_quadexpanded():
    dataMean(0,2)
    return

def test_sfVariance_static_quadexpanded():
    sfVariance(0,2)
    return

def test_sfOrder_static_quadexpanded():
    sfOrder(0,2)
    return

def test_dataVariance_static_cubeexpanded():
    dataVariance(0,3)
    return

def test_dataMean_static_cubeexpanded():
    dataMean(0,3)
    return

def test_sfVariance_static_cubeexpanded():
    sfVariance(0,3)
    return

def test_sfOrder_static_cubeexpanded():
    sfOrder(0,3)
    return
