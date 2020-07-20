import numpy as np
import scipy.linalg as LA
from scipy import stats
import matplotlib.pyplot as plt

import TEP_Import as imp
from DataNode import IncrementalNode
from Norm import IncrementalNorm

import warnings
import itertools
from math import factorial as fac

class IncSFA(IncrementalNode):
    '''
    Incremental slow feature analysis class that takes a set of input signals 
    and calculates slow features and monitoring statistics
    '''
    t = 0         # Time index

    I = 0         # Number of variables
    J = 0         # Number of slow features
    K = 0         # Number of principal component vectors
    L = 0         # Amnesiac parameter

    W = None      # Slow feature transformation matrix
    v_gam = None  # First PC in whitened difference space

    delta = 1     # Sampling interval
    z_prev = None # Previous normalized input
    z_curr = None # Current normalized input

    y_prev = None # Previous slow feature output
    y_curr = None # Current slow feature output
    
    def __init__(self,num_variables, J, K, theta = None,
                 expansion_order = 1, dynamic_copies = 0, L = 3):
        '''
        Constructor for class which takes in desired dimensions, scheduling
        paramters and preprocessing parameters
        '''
        super().__init__(num_variables, dynamic_copies, expansion_order)
        self._store_theta(theta)
        I = self.processed_num_variables
        
        if (J <= K and K <= I):
            self.I = I
            self.J = J
            self.K = K
        else:
            raise RuntimeError("Dimensions must follow J <= K <= I" +
                               ": J = " + str(J) +
                               ", K = " + str(K) +
                               ", I = " + str(I))

        self.cutoff = int(J/2)
        self.L = L
        # Initialize matrices to random normal values
        self.W = np.random.randn(K,J)
        self.v_gam = np.random.randn(K,1)

        self.Lambda = np.random.randn(J)
        self.perm   = np.eye(J,J) # Permutation matrix
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

    def _CIMCA_update(self,W,J,z_dot,gamma,eta):
        '''
        Updates current minor component matrix W with new data point z_dot
        and parameters gamma and eta
        '''
        z_dot = z_dot.reshape((self.K))
        gamma = gamma.reshape((self.K))
        L = 0
        for i in range(J):
            prev_col = W[:,i]
            # Find new minor component
            prev = (1 - eta) * prev_col
            new = eta * ( np.dot(z_dot, prev_col) * z_dot + L )
            col_update = prev - new
            
            # Store minor component
            W[:,i] = col_update/LA.norm(col_update)

            # Update competition from lower components
            lower_sum = np.zeros((self.K))
            wi = W[:,i]
            for j in range(i):
                wj = W[:,j]
                lower_sum += np.dot(wj, wi) * wj
                
            L = gamma*lower_sum

        return(W)
    
    def update_monitoring_stats(self, Lambda, y_dot, eta):
        '''
        Estimate the eigenvalues of the transformation matrix
        Lambda -> Previous estimate of eigenvalues
        y_dot  -> Estimated derivative of slow features
        eta    -> Learning rate
        '''
        # TODO: Can I just find the eigenvalues from W directly?
        lam = np.multiply(y_dot, y_dot).reshape(self.J)
        Lambda = (1-eta) * Lambda + eta * lam
        return(Lambda)

    def calculate_monitoring_stats(self, Lambda, y, y_dot, cutoff):
        '''
        Calculate the monitoring statistics
        Lambda  -> Estimated eigenvalues of transformation matrix
        y       -> Current slow features
        y_dot   -> Slow feature derivative
        cutoff  -> Where to split the "slow" and "fast" features
        T_sqr   -> Monitoring stat 1
        T_e_sqr -> Monitoring stat 2
        S_sqr   -> Monitoring stat 3
        S_e_sqr -> Monitoring stat 4
        '''
        y = y.reshape((self.J,1))
        y_dot = y_dot.reshape((self.J,1))

        # Split features, derivatives, and speeds into slowest and fastest
        y_slow = y[:cutoff,:]
        y_fast = y[cutoff:,:]
        
        y_dot_slow = y_dot[:cutoff,:]
        y_dot_fast = y_dot[cutoff:,:]

        speed_slow = (Lambda[:cutoff]).reshape((cutoff))
        speed_fast = (Lambda[cutoff:]).reshape((self.J-cutoff))

        # Calculate monitoring stats
        T_sqr = y_slow.T @ y_slow
        T_e_sqr = y_fast.T @ y_fast

        S_sqr =   y_dot_slow.T @ np.diag(speed_slow**-1) @ y_dot_slow
        S_e_sqr = y_dot_fast.T @ np.diag(speed_fast**-1) @ y_dot_fast

        return(T_sqr, T_e_sqr, S_sqr, S_e_sqr)

    def update_permutation_matrix(self):
        '''
        Update the order of the slow features based on their current speeds
        '''
        order = self.Lambda.argsort()
        self.Lambda = self.Lambda[order]
        self.perm = self.perm[order,:]
        self.y_prev = self.y_prev @ self.perm
        return

    def add_data(self,raw_signal,update_monitors=True,calculate_monitors=False):
        '''
        Function used to update the IncSFA model with new data points
        Outputs the slow features for the signal input
        raw_signal         -> The data point used to update the model and calculate the features
        calculate_monitors -> Calculate and output the T, T_e, S and S_e stats
        '''
        # Default values
        y = np.zeros((self.J,1))
        stats = [0,0,0,0]

        ''' Signal preprocessing '''
        raw_signal = raw_signal.reshape((raw_signal.shape[0], 1))
        x, is_full = self.process_signal(raw_signal)
        if not is_full:
            # Need more data to get right number of dynamic copies
            return(y, stats)
        
        ''' Update Learning rates '''
        eta_PCA, eta_MCA = self._learning_rate_schedule(self.theta, self.t)

        ''' Signal centering and whitening '''
        if self.t > 0:
            # Updates normalized signals stored
            self.z_prev = np.copy(self.z_curr)
        else:
            # On first pass through, create the norm object
            self.IncNormObject = IncrementalNorm(x, self.K, self.L)
        self.z_curr = self.IncNormObject.normalize_online(x, eta_PCA)

        ''' Transformation '''
        # t > 0, derivative can be calculated, proceed to update model
        if self.t > 0:
            z_dot = (self.z_curr - self.z_prev)/self.delta
            # Update first PC in whitened difference space and gamma
            ''' Derivative centering and first eigenvector output '''
            if self.t == 1:
                self.DerivativeIncNormObject = IncrementalNorm(z_dot, 1, self.L)
            self.v_gam = self.DerivativeIncNormObject.update_CCIPA(z_dot, eta_PCA)
            gam = self.v_gam/LA.norm(self.v_gam)

            # Updates minor components
            self.W = self._CIMCA_update(self.W,self.J,z_dot,gam,eta_MCA)

            y =  (self.z_curr.T @ self.W) @ self.perm 

            self.y_prev = np.copy(self.y_curr)
            self.y_curr = y
        
        ''' Update monitoring stats '''
        if self.t > 2 and (update_monitors or calculate_monitors):
            # Approximate derivative
            y_dot = (self.y_curr - self.y_prev)/self.delta
            if update_monitors:
                # Update the eigenvalue estimate
                self.Lambda = self.update_monitoring_stats(self.Lambda, y_dot, eta_PCA)
            if calculate_monitors:
                # Calculate the monitoring stats
                stats = self.calculate_monitoring_stats(self.Lambda, y, y_dot, self.cutoff)
        
        ''' Update model and return '''
        self.t += 1 # Increase time index
        return(y, stats)

    def evaluate(self, raw_signal, alpha):
        '''
        Function used to test the IncSFA model with new data points
        Outputs the slow features for the signal input
        raw_signal         -> The data point used to test the model and calculate the features
        '''
        # Default values
        y = np.zeros((self.J,1))
        stats = [0,0,0,0]
        stats_crit = [0,0,0,0]

        ''' Signal preprocessing '''
        raw_signal = raw_signal.reshape((raw_signal.shape[0], 1))
        x, is_full = self.process_signal(raw_signal)
        if not is_full:
            raise RuntimeError("Not enough data has been passed to the model to test it")

        ''' Signal centering and whitening '''
        self.z_prev = np.copy(self.z_curr)
        self.z_curr = self.IncNormObject.normalize_similar(x)

        ''' Transformation '''
        self.y_prev = np.copy(self.y_curr)
        y = (self.z_curr.T @ self.W) @ self.perm
        self.y_curr = y
        
        # Approximate derivative
        y_dot = (self.y_curr - self.y_prev)/self.delta
        
        # Calculate the monitoring stats
        stats = self.calculate_monitoring_stats(self.Lambda, y, y_dot, self.cutoff)
        stats_crit = self.calculate_crit_values(alpha)

        return(y, stats, stats_crit)

    def _learning_rate_schedule(self, theta, t):
        '''
        Sets the rate at which the model updates based on parameters theta
        and current time index
        '''
        t1, t2, c, r, nl, nh, T = theta
        t += 1 # Prevents divide by 0 errors
        if t == 1:
            mu_t = -1e-6
        elif t <= t1:
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

    def calculate_crit_values(self,alpha=0.01):
        '''
        Calculate critical values for monitoring
        '''
        if alpha > 1:
            raise ValueError("Confidence level is capped at 1")
        p = 1 - alpha
        N  = self.t
        Md = self.cutoff
        Me = self.J - self.cutoff
        gd = (Md*(N**2-2*N))/((N-1)*(N-Md-1))
        ge = (Me*(N**2-2*N))/((N-1)*(N-Me-1))

        T_d_crit = stats.chi2.ppf(p,Md)
        T_e_crit = stats.chi2.ppf(p,Me)
        S_d_crit = gd*stats.f.ppf(p,Md,N-Md-1)
        S_e_crit = ge*stats.f.ppf(p,Me,N-Me-1)

        crit_vals = [T_d_crit, T_e_crit, S_d_crit, S_e_crit]

        return(crit_vals)

def run(theta, epochs, fignum = 0, figdpi = 200, plot_features = 5, plot_last_epoch = True, alert = True):
    # Import data
    training_sets = list(imp.importTrainingSets([0]))
    testing_sets  = list(imp.importTestSets([4,5,10]))

    # Delete ignored variables
    ignored_var = list(range(22,41))

    X = training_sets[0][1]
    T4  = testing_sets[0][1].T
    T5  = testing_sets[1][1].T
    T10 = testing_sets[2][1].T
    data_points = X.shape[1]

    # Remove unwanted variables from data
    ignored_var = list(range(22,41))
    X = np.delete(X,ignored_var,axis=0)
    T4 = np.delete(T4,ignored_var,axis=0)
    T5 = np.delete(T5,ignored_var,axis=0)
    T10 = np.delete(T10,ignored_var,axis=0)

    # IncSFA parameters
    num_vars = X.shape[0]
    K = 20
    J = 20
    cutoff = 10
    d = 2    # Lagged copies
    n = 1    # Expansion order

    # Create IncSFA object
    SlowFeature = IncSFA(num_vars,J,K,theta,n,d)
    SlowFeature.delta = 3
    SlowFeature.cutoff = cutoff

    # Create empty arrays to store outputs
    total_data_points = data_points * epochs
    Y = np.zeros( (J, total_data_points) )

    stats = np.zeros( (4, total_data_points) )
    stats_crit = np.zeros( (4, total_data_points) )
    
    # Train model
    for j in range(epochs):
        print("Running epoch " + str(j+1) + "/" + str(epochs))
        for i in range(X.shape[1]):
            pos = j*X.shape[1]+i
            run = SlowFeature.add_data(X[:,i])

            # Store data
            Y[:,pos] = run[0].reshape((J))
            stats[:,pos] = run[1]

    # Test data
    test_data = [("IDV(4)",T4),("IDV(5)",T5),("IDV(10)",T10)]
    num_tests = len(test_data)
    col_pos = 1
    monitors_figname = "Monitors" + str(fignum)
    _m = plt.figure(monitors_figname)
    plt.subplots_adjust(wspace=0.4)    
    
    for name, test in test_data:
        data_points = test.shape[1]
        stats = np.zeros( (4, data_points) )
        stats_crit = np.zeros( (4, data_points) )
        
        for i in range(data_points):
            _, stats[:,i], stats_crit[:,i]  = SlowFeature.evaluate(test[:,i], alpha = 0.05)
        
        # Plotting
        plt.subplot(4,num_tests,col_pos)
        plt.title(name)
        plt.plot(stats[0])
        plt.plot(stats_crit[0])
        plt.ylabel("$T^2$")
        plt.xticks([])

        plt.subplot(4,num_tests,col_pos+num_tests)
        plt.plot(stats[1])
        plt.plot(stats_crit[1])
        plt.ylabel("$T^2_e$")
        plt.xticks([])

        plt.subplot(4,num_tests,col_pos+num_tests*2)
        plt.plot(stats[2])
        plt.plot(stats_crit[2])
        plt.ylabel("$S^2$")
        plt.xticks([])

        plt.subplot(4,num_tests,col_pos+num_tests*3)
        plt.plot(stats[3])
        plt.plot(stats_crit[3])
        plt.ylabel("$S^2_e$")
        plt.xlabel("Sample")

        col_pos += 1
    _m.set_size_inches(16, 9)
    plt.savefig(monitors_figname,dpi=figdpi)
    plt.close(fig=_m)

    # Forget everything except last epoch
    if plot_last_epoch:
        Y = Y[:,-data_points:]
        stats = stats[:,-data_points:]
    
    ''' Plotting '''
    figure_text = ( "Epochs: " + str(epochs) + " | $t_1=$ " + str(theta[0]) +
                    " | $t_2=$ " + str(theta[1]) + " | $c=$ " + str(theta[2]) +
                    " | $r=$ " + str(theta[3]) + " | $\eta_l=$ " + str(theta[4]) +
                    " | $\eta_h=$ " + str(theta[5]) + " | $T=$ " + str(theta[6]) +
                    " | $K=$ " + str(K) + " | $J=$" + str(J) + " | Lagged Copies=" +
                    str(d) + " | Expansion Order=" + str(n))
    # Plot Stats
    stats_figname = "Stats" + str(fignum)
    _s = plt.figure(stats_figname)
    plt.figtext(0.25,0.05,figure_text)
    titles = ["$T^2$", "$T_e^2$", "$S^2$", "$S_e^2$"]
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title(titles[i])
        plt.xlabel("Sample")
        plt.plot(stats[i,:])
    _s.set_size_inches(16, 9)
    plt.savefig(stats_figname,dpi=figdpi)
    plt.close(fig=_s)

    # Plot features
    features_figname = "Features" + str(fignum) 
    _f = plt.figure(features_figname)
    plt.figtext(0.25,0.05,figure_text)

    # Calculate eta values
    eta = np.around((Y.shape[1]/(2*np.pi)) * np.sqrt(SlowFeature.Lambda),2)

    mid = int(Y.shape[0]/2)
    slowest = Y[:plot_features,:]
    middle = Y[mid:mid+plot_features,:]
    fastest = Y[-plot_features:,:]

    speed_slowest = eta[:plot_features]
    speed_middle = eta[mid:mid+plot_features]
    speed_fastest = eta[-plot_features:]

    for i in range(plot_features):
        plt.subplot(plot_features,3,3*i+1)
        plt.plot(slowest[i,:], label="$\eta$: " + str(speed_slowest[i]))
        plt.legend(loc="lower left")
        if i == 0:
            plt.title("Slowest")
            
        plt.subplot(plot_features,3,3*i+2)
        plt.plot(middle[i,:], label="$\eta$: " + str(speed_middle[i]))
        plt.legend(loc="lower left")
        if i == 0:
            plt.title("Middle")
            
        plt.subplot(plot_features,3,3*i+3)
        plt.plot(fastest[i,:], label="$\eta$: " + str(speed_fastest[i]))
        plt.legend(loc="lower left")
        if i == 0:
            plt.title("Fastest")
    _f.set_size_inches(16, 9)
    plt.savefig(features_figname,dpi=figdpi)
    plt.close(fig=_f)
    
    if alert:
        print("\a\a") # Ping user once run is complete
  
if __name__ == "__main__":
    np.random.seed(1)
    epochs = 25
    t1 = [10]
    t2 = [100]
    c  = [4]
    r  = [100]
    nl = [0.08]
    nh = [0.08]
    T = [200]
    theta = [t1, t2, c, r, nl, nh, T]
    parameter_list = list(itertools.product(*theta))

    for i, theta in enumerate(parameter_list):
        print(str(i+1).rjust(3,'0') + "/" + str(len(parameter_list)) + ": Running ", str(theta))
        run(theta, epochs, str(i).rjust(3,'0'), plot_last_epoch=True, plot_features=5)