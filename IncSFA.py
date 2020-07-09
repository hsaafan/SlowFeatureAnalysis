import numpy as np
from numpy import linalg as LA
from math import factorial as fac
import matplotlib.pyplot as plt
from SFAClass import SFA

import itertools
import TEP_Import as imp
from DataNode import Node

import warnings

eps = 1e-16

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
    z_dot_bar = 0
    x_bar = 0                     # Current mean of input data
    delta = 1                     # Sampling interval
    z_prev = None                 # Previous normalized input
    z_curr = None                 # Current normalized input
    prev_signals = np.empty((0,)) # Previous signals used for dynamizing

    y_prev = None
    y_curr = None

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

        
        self.V = np.random.randn(I,K)
        self.W = np.random.randn(K,J)
        self.v_gam = np.zeros((K,1)) + eps
        self.z_dot_bar = np.zeros((K,1))
        # Normalize matrix columns
        for col in range(K):
            self.V[:,col] = self.V[:,col]/LA.norm(self.V[:,col])
        for col in range(J):
            self.W[:,col] = self.W[:,col]/LA.norm(self.W[:,col])
        
        self.Lambda = np.ones((J,1))
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
            if not(theta[4] <= theta[5]):
                raise RuntimeError("Values must be eta_l <= eta_h") 
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
        u = u.reshape((u.shape[0],1))

        for i in range(K):
            col = np.copy(V[:,i]).reshape((V.shape[0],1))
            col_norm = LA.norm(col)
            
            # Find new principal component
            prev = (1-eta)*col
            new = eta*(np.matmul(u.T,col)/col_norm)*u

            col_new = prev + new
            col_new_norm = LA.norm(col_new)
            col_normed = col_new/col_new_norm
            V[:,i] = col_new.reshape((V.shape[0]))
            
            # Update residuals
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
            inv_square_v[i] = (col_norm + 1e-18) ** (-1/2)


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
    
    def monitoring_stats(self, y, y_dot, eta, cuttoff):
        y = y.reshape((self.J,1))
        y_dot = y_dot.reshape((self.J,1))
        lam = np.multiply(y_dot, y_dot)
        # Update approximation
        self.Lambda = (1-eta) * self.Lambda + eta * lam
        T_sqr = np.matmul(y[:cuttoff,:].T, y[:cuttoff,:])
        T_e_sqr = np.matmul(y[cuttoff:,:].T, y[cuttoff:,:])
        features_slow = (self.Lambda[:cuttoff]).reshape((cuttoff))
        features_fast = (self.Lambda[cuttoff:]).reshape((self.J-cuttoff))
        S_sqr = np.matmul(np.matmul(y_dot[:cuttoff,:].T,np.diag(features_slow**-1)),y_dot[:cuttoff,:])
        S_e_sqr = np.matmul(np.matmul(y_dot[cuttoff:,:].T,np.diag(features_fast**-1)),y_dot[cuttoff:,:])
        return(T_sqr, T_e_sqr, S_sqr, S_e_sqr)

    def add_data(self,raw_signal):
        '''
        Function used to update the IncSFA model with new data points
        Outputs the slow features for the signal input
        '''
        
        # Default values
        y = np.zeros((self.J))
        stats = [0,0,0,0]

        # Dynamization
        raw_signal = self._dynamize(raw_signal)
        if raw_signal.shape[0] < self.I_dyn:
            # Return a default value if not enough lagged copies available
            return(y, stats)
        
        # Nonlinear expansion
        raw_signal = raw_signal.reshape(raw_signal.shape[0],1)
        x = self.nonlinear_expansion(raw_signal,self.expansion_order)

        # Get new learning rates according to schedule
        eta_PCA, eta_MCA = self._learning_rate_schedule(self.theta, self.t)

        # Mean estimation and centering
        if self.t == 0:
            self.x_bar = x + eps
            self.x_std = np.abs(x) + eps
        else:
            self.x_bar = (1-eta_PCA)*self.x_bar + eta_PCA*x
        u = (x - self.x_bar)/self.x_std

        # Update principal components and create whitening matrix
        self.V = self.CCIPA(self.V,self.K,u,eta_PCA)
        S = self.WhiteningMatrix(self.V)
        
        # Updates normalized signals stored
        if self.t > 0:
            self.z_prev = np.copy(self.z_curr)
        self.z_curr = S @ u


        # t > 0, derivative can be calculated, proceed to update model
        if self.t > 0:
            z_dot = (self.z_curr - self.z_prev)/self.delta
            
            # Update first PC in whitened difference space and gamma
            self.v_gam = self.CCIPA(self.v_gam, 1, z_dot, eta_PCA)
            gam = self.v_gam/LA.norm(self.v_gam)

            # Updates minor components
            self.W = self._CIMCA_update(self.W,self.J,z_dot,gam,eta_MCA)
            y = self.z_curr.T @ self.W
            self.y_prev = np.copy(self.y_curr)
            self.y_curr = np.copy(y)
        
        if self.t > 2:
            y_dot = (self.y_curr - self.y_prev)/self.delta
            stats = self.monitoring_stats(self.y_curr, y_dot, eta_PCA, self.cuttoff)
        self.t += 1 # Increase time index
        
        return(y, stats)

    def _learning_rate_schedule(self, theta, t):
        '''
        Sets the rate at which the model updates based on parameters theta
        and current time index
        '''
        t1, t2, c, r, nl, nh, T = theta

        t += 1
        
        if t <= t1:
            mu_t = -0.5
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

    def CCIPA(self, V, K, x, eta):
        max_iter = int(np.min([K,self.t]))
        u = np.empty((self.I, max_iter + 1))
        u[:,0] = x.reshape((self.I))
        for i in range(max_iter):
            if i == self.t:
                V[:,i] = u[:,i]
            else:
                V_col = V[:,i].reshape((self.I,1))
                u_col = u[:,i].reshape((self.I,1))
                
                prev = (1-eta)*V_col
                new = eta * (u_col @ u_col.T) @ (V_col / LA.norm(V_col))
                V_col_new = prev + new
                V_col_new_normed = V_col_new/ LA.norm(V_col_new)
                V[:,i] = (V_col_new).reshape((self.I))
                u[:, i+1] = (u_col - (u_col.T @  V_col_new_normed ) * V_col_new_normed).reshape((self.I))

        return(V)

    def WhiteningMatrix(self,V):
        D = np.empty((V.shape[1]))
        for i in range(V.shape[1]):
            lam = LA.norm(V[:,i])
            V[:,i] = V[:,i] / lam
            D[i] = (lam + eps) ** (-1/2)
        
        S = V @ np.diag(D) @ V.T

        return(S)

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


def run(theta, epochs, fignum = 0, figdpi = 450, plot_features = 5, plot_last_epoch = True):

    training_sets = list(imp.importTrainingSets([0]))
    X   = training_sets[0][1]
    ignored_var = list(range(22,41))
    X = np.delete(X,ignored_var,axis=0)
    d = 2    # Lagged copies
    q = 0.1  # Partition fraction
    n = 1    # Expansion order

    # Run SFA
    num_vars = 33
    K = 99
    J = 99
    cuttoff = 55

    SlowFeature = IncSFA(num_vars,J,K,theta,n,d)
    SlowFeature.delta = 3
    SlowFeature.cuttoff = cuttoff

    Y = np.zeros((cuttoff,X.shape[1]*epochs))
    stats = np.zeros((4,X.shape[1]*epochs))
    
    for j in range(epochs):
        print("Running epoch " + str(j+1) + "/" + str(epochs))
        for i in range(X.shape[1]):
            run = SlowFeature.add_data(X[:,i])
            Y[:,j*X.shape[1]+i] = (run[0]).reshape((J))[:cuttoff]
            stats[:,j*X.shape[1]+i] = run[1]
            
    if plot_last_epoch:
        start = -X.shape[1]
        Y = Y[:,start:]
        stats = stats[:,start:]
    
    ''' Plotting '''
    figure_text = ( "Epochs: " + str(epochs) + " | $t_1=$ " + str(theta[0]) +
                    " | $t_2=$ " + str(theta[1]) + " | $c=$ " + str(theta[2]) +
                    " | $r=$ " + str(theta[3]) + " | $\eta_l=$ " + str(theta[4]) +
                    " | $\eta_h=$ " + str(theta[5]) + " | $T=$ " + str(theta[6]) )
    # Plot Stats
    stats_figname = "Stats" + str(fignum)
    _s = plt.figure(stats_figname)
    plt.figtext(0.05,0.05,figure_text)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(stats[i,:])
    plt.savefig(stats_figname,dpi=figdpi)
    plt.close(fig=_s)

    # Plot features
    features_figname = "Features" + str(fignum) 
    _f = plt.figure(features_figname)
    plt.figtext(0.05,0.05,figure_text)

    mid = int(Y.shape[0]/2)
    slowest = Y[:plot_features,:]
    middle = Y[mid:mid+plot_features,:]
    fastest = Y[-plot_features:,:]
    
    for i in range(plot_features):
        plt.subplot(plot_features,3,3*i+1)
        plt.plot(slowest[i,:])
        if i == 0:
            plt.title("Slowest")
            
        plt.subplot(plot_features,3,3*i+2)
        plt.plot(middle[i,:])
        if i == 0:
            plt.title("Middle")
            
        plt.subplot(plot_features,3,3*i+3)
        plt.plot(fastest[i,:])
        if i == 0:
            plt.title("Fastest")
    plt.savefig(features_figname,dpi=figdpi)
    plt.close(fig=_f)
    print("\a")

    
if __name__ == "__main__":
    epochs = 5
    t1 = [50]
    t2 = [200]
    c  = [4]
    r  = [1000]
    nl = [0.05]
    nh = [0.1]
    T = [100, 1000]

    theta = [t1, t2, c, r, nl, nh, T]
    parameter_list = list(itertools.product(*theta))

    for i, theta in enumerate(parameter_list):
        print(str(i+1).rjust(3,'0') + "/" + str(len(parameter_list)) + ": Running ", str(theta))
        run(theta, epochs, str(i).rjust(3,'0'))
    
    ''' Proof of concept from paper
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
    '''
