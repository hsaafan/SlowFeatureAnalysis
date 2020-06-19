import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import TEP_Import as imp
from DataNode import Node

import warnings

class IncSFA(Node):
    '''
    Incremental slow feature analysis class that takes a set of input signals 
    and calculates slow features and monitoring statistics
    '''
    t = 1                         # Time index
    I = 0                         # Number of variables
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
    
    def __init__(self,I,J,K,theta = None,expansion_order = None,
                 dynamic_copies = None):
        '''
        Constructor for class which takes in desired dimensions, scheduling
        paramters and preprocessing parameters
        '''
        if not(J <= K and K <= I):
            raise RuntimeError("Dimensions must follow J <= K <= I")
        else:
            self.I = I
            self.J = J
            self.K = K
        # TODO: Check how to initialize these
        self.V = np.eye(I,K)
        self.W = np.eye(I,J)
        self.v_gam = np.eye(I,1)

        self._store_dynamic_copies(dynamic_copies)
        self._store_expansion_order(expansion_order)
        self._store_theta(theta)
        
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
            if not(theta[0] < theta[1] and theta[1] < theta[6]):
                raise RuntimeError("Values must be t1 < t2 < T")
            if not(theta[4] < theta[5]):
                raise RuntimeError("Values must be eta_l < eta_h") 
            self.theta = theta
        else:
            raise RuntimeError("Expected 7 parameters in order: " +
                               "t1, t2, c, r, eta_l, eta_h, T")
        return
    
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
            col = V[:,i]
            col_norm = LA.norm(col)
            V_hat[:,i] = col/col_norm
            # Get the inverse square of the norm of each column
            inv_square_v[i] = col_norm ** (-1/2)
        D = np.diag(inv_square_v)
        S = np.matmul(V_hat,D)
        return(S)

    def _CCIPCA_update(self, V, K, u, eta):
        '''
        Updates current principal component matrix with new sample u
        and parameter eta
        '''
        for i in range(K):
            col = V[:,i].reshape((self.I,1))
            col_norm = LA.norm(col)
            # Find new principal component
            prev = (1-eta)*col
            new = eta*(float(np.matmul(u.T,col))/col_norm)*u
            col_new = prev + new
            col_new_norm = LA.norm(col_new)
            V[:,i] = col_new.reshape((self.I))
            # TODO: Check this tolerance value
            if col_new_norm < 1e-12:
                # For sufficiently small principal components, set residual
                # matrix to 0
                u = np.zeros_like(u)
            else:
                # Update residuals
                col_normed = col_new/col_new_norm
                u = u - np.matmul(u.T,col_normed)*col_normed
        return(V)

    def _CIMCA_update(self,W,J,z_dot,gamma,eta):
        '''
        Updates current minor component matrix W with new data point z_dot
        and parameters gamma and eta
        '''
        L = 0
        for i in range(J):
            col = W[:,i].reshape((self.I,1))
            # Find new minor component
            prev = (1-eta)*col
            new = eta*(float(np.matmul(z_dot.T,col))*z_dot+L)
            col_update = (prev - new).reshape((self.I))
            
            # Store minor component
            W[:,i] = col_update/LA.norm(col_update)

            # Update competition from lower components
            lower_sum = 0
            wi = W[:,i].reshape((self.I,1))
            for j in range(i):
                wj = W[:,j].reshape((self.I,1))
                lower_sum += float(np.matmul(wj.T,wi))*wj
            L = gamma*lower_sum
        return(W)

    def add_data(self,raw_signal):
        '''
        Function used to update the IncSFA model with new data points
        Outputs the slow features for the signal input
        '''
        # Default values
        finished = False
        y = np.zeros((1,self.J))

        # Dynamizes the raw_signal
        if self.dynamic:
            d = self.dynamic_copies
            lagged_copies_length = self.I - int(self.I/(d+1))
            # Checks if there are enough previous signals to proceed
            if self.prev_signals.shape[0] == lagged_copies_length:
                # Adds lagged copies to signal and updates prev_signals
                raw_signal = np.append(raw_signal,self.prev_signals)
                self.prev_signals = raw_signal[:lagged_copies_length]
            else:
                # Updates prev_signals and return
                self.prev_signals = np.append(raw_signal,self.prev_signals)
                finished = True

        # Main model updating
        if not finished:
            raw_signal = raw_signal.reshape((raw_signal.shape[0],1))
            x = self.nonlinear_expansion(raw_signal,self.expansion_order)
            # Get new learning rates according to schedule
            eta_PCA, eta_MCA = self._learning_rate_schedule(self.theta, self.t)

            # Updates the variable means used
            self.x_bar = (1-eta_PCA)*self.x_bar+eta_PCA*x
            
            # Center the signal
            u = x - self.x_bar
            
            # Update principal components and create whitening matrix
            self.V = self._CCIPCA_update(self.V,self.K,u,eta_PCA)
            S = self._construct_whitening_matrix(self.V)
            # Updates normalized signals stored
            if self.t > 1:
                self.z_prev = self.z_curr
            self.z_curr = np.matmul(S.T,u)

            # If t > 1, derivative can be calculated, proceed to update model
            if self.t > 1:
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
        # Get parameters from theta
        t1, t2, c, r, eta_l, eta_h, T = theta

        # Update eta_PCA
        if t == 1:
            mu_t = -0.5
        elif t <= t1:
            mu_t = 0
        elif t <= t2:
            mu_t = c*(t-t1)/(t2-t1)
        elif t2 < t:
            mu_t = c + (t-t2)/r
        eta_PCA = (1+mu_t)/t

        # Update eta_MCA
        if t <= T:
            eta_MCA = eta_l + (eta_h - eta_l) * (t/T) ** 2
        else:
            eta_MCA = eta_h
        return(eta_PCA,eta_MCA)


if __name__ == "__main__":
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
    theta = [10,200,3,500,0,0.01,500] # Learning schedule parameters

    # Run SFA
    I = X.shape[0]*(d+1)
    K = I
    J = 55
    SlowFeature = IncSFA(I,J,K,theta,n,d)
    SlowFeature.delta = 3
    Y = np.zeros((J,X.shape[1]))
    Z = np.zeros((K,X.shape[1]))
    for i in range(X.shape[1]):
        Y[:,i] = SlowFeature.add_data(X[:,i])
        if not SlowFeature.z_curr is None:
            Z[:,i] = SlowFeature.z_curr.reshape(I)
    Z = Z[:,-100:]
    print(np.std(Z,axis=1))
    print(np.mean(Z,axis=1))
    Y = Y[:,-100:]
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
