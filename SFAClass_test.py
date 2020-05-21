import numpy as np
from SFAClass import SFA

PREC = 12 # Consider any number less than 10^(-PREC) to be 0


def dataSetup(length, delayed_copies):
    # Setting up a data sample
    # Example taken from:
    # https://towardsdatascience.com/a-brief-introduction-to-slow-feature-analysis-18c901bc2a58
    
    S = np.zeros((length,1),'d')
    D = np.zeros((length,1),'d')
    S[0] = 0.6
    for t in range(1,length):
        D[t] = np.sin(np.pi/75. * t) - t/150.
        S[t] = (3.7+0.35*D[t]) * S[t-1] * (1 - S[t-1])
                                           
    X = np.zeros((length-(delayed_copies-1),delayed_copies),'d')
    for i in range(0,delayed_copies):
       X[:,i] = S[i:length+i+1-delayed_copies,0]

    return X

def objectSetup():
    X = dataSetup(300,4)
    SlowFeature = SFA(X)
    SlowFeature.train()
    return SlowFeature


def test_dataVariance():
    SlowFeature = objectSetup()
    Z = SlowFeature.normalized_expanded_data
    cov_matrix = np.matmul(Z,Z.T)

    num_features = Z.shape[0]
    ident = np.identity(num_features)

    diff = np.around(cov_matrix - ident,PREC)
    
    assert not np.all(diff)
    return

def test_dataMean():
    SlowFeature = objectSetup()
    Z = SlowFeature.normalized_expanded_data
    Zmeans = Z.mean(axis=1)

    assert not np.all(np.around(Zmeans,PREC))
    return

def test_sfVariance():
    SlowFeature = objectSetup()
    Y = SlowFeature.slow_features
    cov_matrix = np.matmul(Y,Y.T)

    num_features = Y.shape[0]
    ident = np.identity(num_features)

    diff = np.around(cov_matrix - ident,PREC)
    
    assert not np.all(diff)
    return

def test_sfOrder():
    SlowFeature = objectSetup()
    Y = SlowFeature.slow_features
    
    Ydot = Y[:,1:] - Y[:,:-1]
    num_features = Y.shape[0]

    precision = 10**(-PREC)
    for i in range(0,num_features):
        for j in range(i+1,num_features):
            prod = np.matmul(Y[i,:],Y[j,:])
            assert abs(prod) < precision
    return

