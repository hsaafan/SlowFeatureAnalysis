'''
Implementation of 2 PCA algorithms

NIPALS uses an iterative method to solve for principle components
SVDPCA uses the SVD method to solve for principle components
'''

import numpy as np
import matplotlib.pyplot as plt

from numpy import matmul
from numpy.linalg import norm
from sklearn import datasets


def nipals(data_matrix,p_comp = 1,conv_criteria=1e-16,max_iter=1000):
    # Scale the data matrix to 0 mean and unit variance
    col_mean = data_matrix.mean(axis=0)
    col_stdv  = data_matrix.std(axis=0)
    scaled_data = (data_matrix-col_mean)/col_stdv
    
    # Initialize empty tensors to store data
    rows = data_matrix.shape[0]
    cols = data_matrix.shape[1]
    X = np.empty((p_comp+1,rows,cols),dtype=float)
    t = np.empty((p_comp,rows,  1 ),dtype=float)
    p = np.empty((p_comp,cols,  1 ),dtype=float)

    # Set starting X matrix
    X[0] = scaled_data
    for i in range(0,p_comp):
        # Loop for as many principle components as chosen

        # Get the current X matrix and choose t as some column of it
        X_curr = X[i]
        t_curr = X_curr[:,1].reshape(rows,1)
        # Set initial values for stopping criterion
        conv_err = np.inf
        iteration = 0
        while conv_err > conv_criteria and iteration < max_iter:
            # Iteratively solve for loadings and scores
            
            # Get current loading vector
            p_curr = matmul(X_curr.T,t_curr)/norm(matmul(X_curr.T,t_curr))

            # Caculate new score vector based on loading vector
            t_prev = t_curr
            t_curr = matmul(X_curr,p_curr)

            # Calculate maximum convergence error and increase iteration
            conv_err = ((t_curr-t_prev)/t_prev)
            conv_err = norm(conv_err)
            iteration += 1

        # Store current score and loading vectors    
        t[i] = t_curr.reshape(rows,1)
        p[i] = p_curr.reshape(cols,1)
        # Residual deflation step
        X[i+1] = X_curr - matmul(t[i],p[i].T)

    # Create score and loading matrices and return
    T = t.reshape(rows,p_comp)
    P = p.reshape(cols,p_comp)
    return(T,P,X,scaled_data)
    

def SVDPCA(data_matrix,k=1):
    # Scale the data to 0 mean and unit variance
    col_mean = data_matrix.mean(axis=0)
    col_stdv  = data_matrix.std(axis=0)
    X = (data_matrix-col_mean)/col_stdv
    
    rows = data_matrix.shape[0]
    cols = data_matrix.shape[1]

    # Find the SVD of data
    [U,Sigma,Vh] = np.linalg.svd(X)

    # Create the diagonal rectangular matrix of singular values
    D = np.zeros((rows,cols))
    D[:k,:k] = np.diag(Sigma[:k])

    # V and P (loadings) are equal
    P = Vh.T

    # T (scores) is equal to U*D
    T = matmul(U,D)

    return(T,P,Sigma)





if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data

    # Eigen decomposition
    [Lambda, W] = np.linalg.eig(matmul(X.T,X))
    print(W)
    W_r = W[:,0:2]
    a1 = matmul(X,W_r)
    
    # SVD Method
    [T,P,Sigma] = SVDPCA(X,4)
    #print("SVD Scores:\n",T)
    print("SVD Loadings:\n",P)
    P_r = P[:,0:2]
    a2 = matmul(X,P_r)

    
    # NIPALS Method
    [t,p,resid,X] = nipals(X,4)
    #print("NIPALS Scores:\n",t)
    print("NIPALS Loadings:\n",p)
    #print("NIPALS Residuals:\n",resid[-1])
    p_r = p[:,0:2]
    a3 = matmul(X,p_r)

    # Plotting data on 2 principal components
    plt.subplot(2,3,1)
    plt.scatter(a1[:,0],a1[:,1])
    plt.title("Eigendecompisition")

    plt.subplot(2,3,2)
    plt.scatter(a2[:,0],a2[:,1])
    plt.title("SVD")

    plt.subplot(2,3,3)
    plt.scatter(a3[:,0],a3[:,1])
    plt.title("NIPALS")

    # Plotting cumulative variance
    plt.subplot(2,3,4)
    x_axis = 1 + np.arange(Lambda.shape[0])
    y_axis = np.cumsum(Lambda)/Lambda.sum()
    plt.step(x_axis,y_axis)

    plt.subplot(2,3,5)
    x_axis = 1 + np.arange(Sigma.shape[0])
    y_axis = np.cumsum(Sigma)/Sigma.sum()
    plt.step(x_axis,y_axis)
    
    plt.show()
