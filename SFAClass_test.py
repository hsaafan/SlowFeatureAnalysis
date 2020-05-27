import numpy as np
from SFAClass import SFA

PREC = 12 # Consider any number less than 10^(-PREC) to be 0


def dataSetup(length):
    # Setting up a data sample
    # Example taken from:
    # https://towardsdatascience.com/a-brief-introduction-to-slow-feature-analysis-18c901bc2a58
    
    S = np.zeros((length,1),'d')
    D = np.zeros((length,1),'d')
    S[0] = 0.6
    for t in range(1,length):
        D[t] = np.sin(np.pi/75. * t) - t/150.
        S[t] = (3.7+0.35*D[t]) * S[t-1] * (1 - S[t-1])
                                           
    X = S.reshape((1,length))
    return X

def objectSetup(k,expand,order):
    X = dataSetup(300)
    SlowFeature = SFA(X)
    if k > 1:
        SlowFeature.dynamize(k+1)
    if expand:
        SlowFeature.expand(order)
    SlowFeature.train()
    return SlowFeature


def dataVariance(k,expand,order):
    SlowFeature = objectSetup(k,expand,order)
    Z = SlowFeature.signals_norm
    cov_matrix = np.matmul(Z,Z.T)

    num_features = Z.shape[0]
    ident = np.identity(num_features)

    diff = np.around(cov_matrix - ident,PREC)
    
    assert not np.all(diff)
    return

def dataMean(k,expand,order):
    SlowFeature = objectSetup(k,expand,order)
    Z = SlowFeature.signals_norm
    Zmeans = Z.mean(axis=1)

    assert not np.all(np.around(Zmeans,PREC))
    return

def sfVariance(k,expand,order):
    SlowFeature = objectSetup(k,expand,order)
    Y = SlowFeature.slow_features
    cov_matrix = np.matmul(Y,Y.T)

    num_features = Y.shape[0]
    ident = np.identity(num_features)

    diff = np.around(cov_matrix - ident,PREC)
    
    assert not np.all(diff)
    return

def sfOrder(k,expand,order):
    SlowFeature = objectSetup(k,expand,order)
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
    dataVariance(4,True,2)
    return

def test_dataMean_dynamic_quadexpanded():
    dataMean(4,True,2)
    return

def test_sfVariance_dynamic_quadexpanded():
    sfVariance(4,True,2)
    return

def test_sfOrder_dynamic_quadexpanded():
    sfOrder(4,True,2)
    return

def test_dataVariance_dynamic_unexpanded():
    dataVariance(4,False,2)
    return

def test_dataMean_dynamic_unexpanded():
    dataMean(4,False,2)
    return

def test_sfVariance_dynamic_unexpanded():
    sfVariance(4,False,2)
    return

def test_sfOrder_dynamic_unexpanded():
    sfOrder(4,False,2)
    return

def test_dataVariance_static_unexpanded():
    dataVariance(1,False,2)
    return

def test_dataMean_static_unexpanded():
    dataMean(1,False,2)
    return

def test_sfVariance_static_unexpanded():
    sfVariance(1,False,2)
    return

def test_sfOrder_static_unexpanded():
    sfOrder(1,False,2)
    return

def test_dataVariance_static_quadexpanded():
    dataVariance(1,True,2)
    return

def test_dataMean_static_quadexpanded():
    dataMean(1,True,2)
    return

def test_sfVariance_static_quadexpanded():
    sfVariance(1,True,2)
    return

def test_sfOrder_static_quadexpanded():
    sfOrder(1,True,2)
    return

def test_dataVariance_static_cubeexpanded():
    dataVariance(1,True,3)
    return

def test_dataMean_static_cubeexpanded():
    dataMean(1,True,3)
    return

def test_sfVariance_static_cubeexpanded():
    sfVariance(1,True,3)
    return

def test_sfOrder_static_cubeexpanded():
    sfOrder(1,True,3)
    return
