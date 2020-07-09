import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import TEP_Import as imp
from DataNode import Node
from Norm import Norm

class RSFA(Node):
    time = 0
    delta = 1   # Time between data inputs
    f_fac = 0.9 # Forgetting factor
    alpha = 0.05 
    cov = None
    cov_dot = None

    prev_u = None
    prev_x = None
    mean_vector = None

    x = None
    x_dot = None
    x_dot_prev = None

    th_1 = None
    th_2 = None
    th_3 = None

    Q = None
    Pd = None
    def __init__(self):
        pass

    def update_mean(self, u):
        self.mean_vector = (1-self.f_fac)*self.mean_vector + self.f_fac*u
        x = u - self.mean_vector
        return(x)

    def update_cov(self,x,x_dot, x_dot_prev):
        self.cov = (1-self.f_fac)*(self.cov+self.mean_vector @ self.mean_vector.T) + self.f_fac*(x @ x.T)
        self.cov_dot = self.f_fac*self.cov_dot + self.f_fac*(1-self.f_fac)*(x_dot_prev @ x_dot.T)
        return

    def decompose_cov(self):
        pass

    def partial_decompose(self):
        pass

    def update_control_limits(self, x, x_dot, alpha):
        # Calculate limits
        c = stats.norm.cdf(alpha)
        t1 = self.th_1
        t2 = self.th_2
        t3 = self.th_3
        h = 1 - (2*t1*t3)/(3*(t2**2))
        Q = t1*(1 + (c*(2*t2*(h**2))**(1/2)/t1) + t2*h*(h-1)/(t1**2)) ** (1/h)

        # Calculate test stats
        ident = np.eye(self.Pd.shape[0],self.Pd.shape[1])
        T_sqr = x.T @ self.Q.T @ self.Pd.T @ self.Pd @ self.Q @ x
        T_e_sqr = x.T @ self.Q.T @ (ident - self.Pd.T @ self.Pd) @ self.Q @ x
        S_sqr = x_dot.T @ self.Q.T @ self.Pd.T @ self.Pd @ self.Q @ x_dot
        return(T_sqr, T_e_sqr, S_sqr, Q)

    def add_data(self, u):
        if u.ndim == 1:
            u = u.reshape((u.shape[0],1))
        if self.prev_u is None:
            # For first iteration
            self.prev_u = u
            self.prev_x = np.zeros_like(u)
            self.mean_vector = u
            return
        
        self.x = self.update_mean(u)
        if self.x_dot is None:
            self.x_dot = (self.x - self.prev_x)/self.delta
            return
        
        self.x_dot_prev = self.x_dot
        self.x_dot = (self.x - self.prev_x)/self.delta

        if self.cov_dot is None:
            self.cov = self.x @ self.x.T
            self.cov_dot = self.x_dot @ self.x_dot.T
        else:
            self.update_cov(self.x,self.x_dot, self.x_dot_prev)
        
        self.decompose_cov()
        self.partial_decompose()
        T_sqr, T_e_sqr, S_sqr, Q = self.update_control_limits(self.x, self.x_dot, self.alpha)

        return(T_sqr, T_e_sqr, S_sqr)

if __name__ == "__main__":
    pass