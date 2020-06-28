import numpy as np
from numpy import linalg as LA
from math import factorial as fac
import matplotlib.pyplot as plt
from SFAClass import SFA

import TEP_Import as imp
from DataNode import Node

import warnings

class IncSFA(Node):
    '''
    Incremental slow feature analysis class that takes a set of input signals 
    and calculates slow features and monitoring statistics
    '''
    t = 0                         # Time index
    I = 0                         # Number of variables
    I_dyn = 0                     # Number of variables pre-expansion
    J = 0                         # Number of slow features
    K = 0                         # Number of principal component vectors
    V = None                      # Principal components
    W = None                      # Slow feature transformation (y = z.T @ W)
    v_gam = None                  # First PC in whitened difference space
    x_bar = 0                     # Current mean of input data
    delta = 1                     # Sampling interval
    z_prev = None                 # Previous normalized input
    z_curr = None                 # Current normalized input
    prev_signals = np.empty((0,)) # Previous signals used for dynamizing

    expansion_order = None        # Order of nonlinear expansion
    dynamic_copies = None         # Number of lagged copies
    dynamic = False               # Whether any lagged copies are used or not
    
    def __init__(self,num_variables, J, K, theta = None,
                 expansion_order = None, dynamic_copies = None):
        '''
        Constructor for class which takes in desired dimensions, scheduling
        paramters and preprocessing parameters
        '''
        self._store_dynamic_copies(dynamic_copies)
        self._store_expansion_order(expansion_order)
        self._store_theta(theta)

        # Get dimension of preprocessed signal
        # From dynamization
        I_dyn = num_variables*(self.dynamic_copies + 1) 
        # From expansion
        # Order 1
        I = I_dyn
        for r in range(2,self.expansion_order+1):
            # Order 2 -> expansion_order
            I += fac(r+I_dyn-1)/(fac(r)*(fac(I_dyn-1)))
        I = int(I)

        if (J <= K and K <= I):
            self.I = I
            self.I_dyn = I_dyn
            self.J = J
            self.K = K
        else:
            raise RuntimeError("Dimensions must follow J <= K <= I" +
                               ": J = " + str(J) +
                               ", K = " + str(K) +
                               ", I = " + str(I))
        
        self.V = np.eye(I,K)
        self.W = np.eye(K,J)
        self.v_gam = np.eye(K,1)

        return

    def _store_dynamic_copies(self,d):
        '''
        Perform checks on number of dynamic copies entered and store value
        '''
        if not (d is None):
            if type(d) == int:
                if d >= 0:
                    self.dynamic = True
                    self.dynamic_copies = d
                elif d == 0:
                    self.dynamic = False
                    self.dynamic_copies = d
                else:
                    raise ValueError("Number of lagged copies must be positive")
            else:
                raise TypeError("Number of lagged copies must be an integer")
        else:
            warnings.warn("Number of lagged copies not specified, using 0",
                          RuntimeWarning)
            self.dynamic = False
            self.dynamic_copies = 0
        return

    def _store_expansion_order(self,n):
        '''
        Perform checks on expansion order and store value
        '''
        if not (n is None):
            if type(n) == int:
                if n >= 1:
                    self.expansion_order = n
                else:
                    raise ValueError("Expansion order must be at least 1")
            else:
                raise TypeError("Expansion order must be an integer")
        else:
            warnings.warn("Expansion order not specified, using 1",
                          RuntimeWarning)
            self.expansion_order = 1
        return

    def _store_theta(self,theta):
        '''
        Perform checks on scheduling parameters and store values
        '''
        if theta is None:
            warnings.warn("No scheduling parameters specified, using " +
                          "example values: t1 = 20, t2 = 200, c = 3, " +
                          "r = 2000, eta_l = 0, eta_h = 0.01", RuntimeWarning)
            self.theta = [20,200,3,2000,0,0.01,2000]
        elif len(theta) == 7:
            if not(theta[0] < theta[1]):
                raise RuntimeError("Values must be t1 < t2")
            if not(theta[4] < theta[5]):
                raise RuntimeError("Values must be eta_l < eta_h") 
            self.theta = theta
        else:
            raise RuntimeError("Expected 7 parameters in order: " +
                               "t1, t2, c, r, eta_l, eta_h, T")
        return

    def _dynamize(self,raw_signal):
        '''
        Adds lagged copies to signal
        '''
        if self.dynamic:
            d = self.dynamic_copies
            n = raw_signal.shape[0]
            lagged_copies_length = d*n
        
            dyn_signal = np.append(raw_signal,self.prev_signals)
            self.prev_signals = dyn_signal[:lagged_copies_length]
        else:
            dyn_signal = raw_signal

        return(dyn_signal)

    def _CCIPCA_update(self, V, K, u, eta):
        '''
        Updates current principal component matrix with new sample u
        and parameter eta
        '''
        u = u.reshape((u.shape[0]))
        if self.t < K:
            V[:,self.t] = u
        else:
            for i in range(K):
                col = np.copy(V[:,i])
                col_norm = LA.norm(col)

                # Find new principal component
                prev = (1-eta)*col
                new = eta*(np.dot(u,col)/col_norm)*u
            
                col_new = prev + new
                col_new_norm = LA.norm(col_new)
                V[:,i] = col_new
            
                # Update residuals
                col_normed = col_new/col_new_norm
                u = u - np.matmul(u.T,col_normed)*col_normed
        return(V)
    
    def _construct_whitening_matrix(self, V):
        '''
        Construcs a whitening matrix W based on principal components V 
        '''
        # Initialize
        V_hat = np.empty_like(V)
        K = V.shape[1]
        inv_square_v = np.empty((K))
        
        for i in range(K):
            # Update columns of V_hat
            col = np.copy(V[:,i])
            col_norm = LA.norm(col)
            V_hat[:,i] = col/col_norm
            # Get the inverse square of the norm of each column
            inv_square_v[i] = col_norm ** (-1/2)
        D = np.diag(inv_square_v)
        S = np.matmul(V_hat,D)
        return(S)

    def _CIMCA_update(self,W,J,z_dot,gamma,eta):
        '''
        Updates current minor component matrix W with new data point z_dot
        and parameters gamma and eta
        '''
        z_dot = z_dot.reshape((self.K))
        gamma = gamma.reshape((self.K))
        if self.t < J:
            W[:,self.t] = z_dot
        else:
            L = 0
            for i in range(J):
                col = np.copy(W[:,i])
                # Find new minor component
                prev = (1-eta)*col
                new = eta*(np.dot(z_dot,col)*z_dot+L)
                col_update = prev - new
            
                # Store minor component
                W[:,i] = col_update/LA.norm(col_update)

                # Update competition from lower components
                lower_sum = np.zeros((self.K))
                wi = np.copy(W[:,i])
                for j in range(i):
                    wj = np.copy(W[:,j])
                    lower_sum += np.dot(wj,wi)*wj
                
                L = gamma*lower_sum
            
        return(W)

    def add_data(self,raw_signal):
        '''
        Function used to update the IncSFA model with new data points
        Outputs the slow features for the signal input
        '''
        
        # Default value
        y = np.zeros((self.J))

        # Dynamization
        raw_signal = self._dynamize(raw_signal)        
        d = self.dynamic_copies
        if raw_signal.shape[0] < self.I_dyn:
            # Return a default value if not enough lagged copies available
            return(y)
        
        # Nonlinear expansion
        raw_signal = raw_signal.reshape(raw_signal.shape[0],1)
        x = self.nonlinear_expansion(raw_signal,self.expansion_order)

        # Get new learning rates according to schedule
        eta_PCA, eta_MCA = self._learning_rate_schedule(self.theta, self.t)

        # Mean estimation and centering
        if self.t == 0:
            self.x_bar = x
            u = x
        else:
            self.x_bar = (1-eta_PCA)*self.x_bar+eta_PCA*x
            u = x - self.x_bar
        
        # Update principal components and create whitening matrix
        self.V = self._CCIPCA_update(self.V,self.K,u,eta_PCA)
        S = self._construct_whitening_matrix(self.V)

        # Updates normalized signals stored
        if self.t > 0:
            self.z_prev = np.copy(self.z_curr)
        self.z_curr = np.matmul(S.T,u)
        
        # t > 0, derivative can be calculated, proceed to update model
        if self.t > 0:
            z_dot = (self.z_curr - self.z_prev)/self.delta
            
            # Update first PC in whitened difference space and gamma
            self.v_gam = self._CCIPCA_update(self.v_gam, 1, z_dot, eta_PCA)
            gam = self.v_gam/LA.norm(self.v_gam)

            # Updates minor components
            self.W = self._CIMCA_update(self.W,self.J,z_dot,gam,eta_MCA)
            y = np.matmul(self.z_curr.T,self.W)
        self.t += 1 # Increase time index
        return(y)

    def _learning_rate_schedule(self, theta, t):
        '''
        Sets the rate at which the model updates based on parameters theta
        and current time index
        '''
        t1, t2, c, r, nl, nh, T = theta

        t += 1
        
        if t <= t1:
            mu_t = 0
        elif t <= t2:
            mu_t = c * (t-t1)/(t2-t1)
        else:
            mu_t = c + (t-t2)/r

        
        eta_PCA = (1+mu_t)/t


        if t <= T:
            eta_MCA = nl + (nh-nl)*((t/T)**2)
        else:
            eta_MCA = nh

            
        return(eta_PCA,eta_MCA)


def data_poc(samples):
    t = np.linspace(0,2*np.pi,num = samples)
    X = np.empty((2,samples))
    X[0,:] = np.sin(t) + np.power(np.cos(11*t),2)
    X[1,:] = np.cos(11*t)
    
    return(X)

def RMSE(epoch_data, true_data):
    J = epoch_data.shape[0]
    T = epoch_data.shape[1]
    RMSE = np.zeros(J)
    for j in range(J):
        RMSE[j] = np.power(np.sum(np.power(epoch_data[j,:] - true_data[j,:],2))/T,0.5)
    return(RMSE)

if __name__ == "__main__":
    '''
    # Import TEP and set up input data
    training_sets = list(imp.importTrainingSets([0]))
    testing_sets  = list(imp.importTestSets([4,5,10]))

    X   = training_sets[0][1]
    T4  = testing_sets[0][1].T
    T5  = testing_sets[1][1].T
    T10 = testing_sets[2][1].T

    # Delete ignored variables
    ignored_var = list(range(22,41))
    X = np.delete(X,ignored_var,axis=0)
    T4 = np.delete(T4,ignored_var,axis=0)
    T5 = np.delete(T5,ignored_var,axis=0)
    T10 = np.delete(T10,ignored_var,axis=0)

    d = 2    # Lagged copies
    q = 0.1  # Partition fraction
    n = 1    # Expansion order
    theta = [20,50,10,300,0,0.008,100] # Learning schedule parameters
    
    # Run SFA
    num_vars = 33
    K = 99
    J = 55
    SlowFeature = IncSFA(num_vars,J,K,theta,n,d)
    SlowFeature.delta = 3
    epochs = 1
    Y = np.zeros((J,X.shape[1]))
    Z = np.zeros((K,X.shape[1]))
    epochs = 10

    for j in range(epochs):
        for i in range(X.shape[1]):
            Y[:,i] = SlowFeature.add_data(X[:,i])
            if not SlowFeature.z_curr is None:
                Z[:,i] = SlowFeature.z_curr.reshape(K)

    #Y = Y[:,-100:]
    mid = int(Y.shape[0]/2)
    slowest = Y[:5,:]
    middle = Y[mid:mid+5,:]
    fastest = Y[-5:,:]
    plt.figure("Slow Features")
    for i in range(5):
        plt.subplot(5,3,3*i+1)
        plt.plot(slowest[i,:])
        if i == 0:
            plt.title("Slowest")
            
        plt.subplot(5,3,3*i+2)
        plt.plot(middle[i,:])
        if i == 0:
            plt.title("Middle")
            
        plt.subplot(5,3,3*i+3)
        plt.plot(fastest[i,:])
        if i == 0:
            plt.title("Fastest")
    plt.show()
    '''

    X = data_poc(500)
    theta = [20,200,4,5000,0,0.01,-1]
    num_vars = 2
    J = 5
    K = 5
    n = 2
    epochs = 120

    data = np.copy(X)
    for j in range(1,epochs):
        if j == 59:
            X_2 = np.empty_like(X)
            x1 = np.copy(X[0,:])
            x2 = np.copy(X[1,:])
            X_2[0,:] = x2
            X_2[1,:] = x1
            data_2 = np.copy(X_2)
            for k in range(60,epochs):
                data_2 = np.concatenate((data_2,X_2),axis=1)
            SlowFeature_switch = SFA(data_2,None,n)
            true_data_2 = SlowFeature_switch.train()
            break
        data = np.concatenate((data,X),axis = 1)
        
    SlowFeature_orig = SFA(data,None,n)

    if epochs > 59:
        true_data = np.concatenate((SlowFeature_orig.train(),true_data_2),
                                   axis=1)
    else:
        true_data = SlowFeature_orig.train()
    
    SlowFeature = IncSFA(num_vars,J,K,theta,n)
    Y = np.zeros((J,X.shape[1]*epochs))
    Z = np.zeros((K,X.shape[1]*epochs))
    err = np.zeros((J,epochs))
    
    for j in range(epochs):
        if j == 59:
            x1 = np.copy(X[0,:])
            x2 = np.copy(X[1,:])
            X[0,:] = x2
            X[1,:] = x1
        
        for i in range(X.shape[1]):
            Y[:,X.shape[1]*j+i] = SlowFeature.add_data(X[:,i])
            if not SlowFeature.z_curr is None:
                Z[:,X.shape[1]*j+i] = SlowFeature.z_curr.reshape(K)

        epoch_data = Y[:,j*X.shape[1]:(j+1)*X.shape[1]]
        epoch_true = true_data[:,j*X.shape[1]:(j+1)*X.shape[1]]
        err[:,j] = RMSE(epoch_data,epoch_true)

    
    for i in range(J):
        plt.subplot(J,1,i+1)
        plt.plot(Y[i,:])
        plt.plot(true_data[i,:])
        #plt.plot(err[i,:])
    plt.show()
