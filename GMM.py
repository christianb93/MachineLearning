#############################################
#
# Gaussian mixtures model with EM
#
# Note that this algorithm has a few weaknesses,
# we do for instance not properly check for singular
# covariance matrices and will therefore sometimes
# abort with an error from scipy
#
# Copyright (c) 2018 christianb93
# Permission is hereby granted, free of charge, to 
# any person obtaining a copy of this software and 
# associated documentation files (the "Software"), 
# to deal in the Software without restriction, 
# including without limitation the rights to use, 
# copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice 
# shall be included in all copies or substantial 
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY 
# OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS 
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#############################################

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import multivariate_normal
import prettytable as pt


##############################################
# Draw from a finite distribution defined
# by a vector p with elements adding up
# to one. We return a number between
# 1 and the number of elements of p
# and i is returned with probability p[i-1]
##############################################
def draw(p):
    u = np.random.uniform()
    x = 0
    n = 1
    for i in p:
        x = x + i
        if x >= u:
            return n
        n += 1
    return i

#############################################
#
# Utility function to create a sample
#
# This will create a sample of data points
# distributed over K clusters. The sample
# will have size points. 
# The clusters are Gaussian normal 
# distributions with the given centres
# and covariance matrices
# 
# The centre points are assumed to be given
# as a matrix with K rows. The vector p is 
# a vector with non-negative elements that
# sum up to one 
# 
# Returns the sample and the true cluster
#############################################

def get_sample(M, p, cov, size=10):
    #
    # Number of clusters
    #
    X = []
    T = []
    for i in range(size):
        k = draw(p) - 1
        T.append(k)
        #
        # Then draw from the normal distribution with mean M[k,:]
        #
        rv = multivariate_normal(mean=M[k], cov=cov[k])
        _X = [rv.rvs()]
        X.append(_X)
    return np.concatenate(X), T
    
    

#############################################
# 
# The K-means algorithm. This is here for
# comparison purposes
#
#############################################

def kMeans(S, K=2, iterations=10):
    #
    # Draw K points from the sample
    # which we use as cluster centres    
    # to build a matrix M of shape (K,d)
    #    
    N = S.shape[0]
    M = S[np.random.randint(low=0, high=N, size=K), :]
    #
    # Run the algorithm
    #
    D = np.zeros(shape=[K])
    for _ in range(iterations):
        #
        # First calculate the assignment matrix R
        #
        R = np.zeros(shape=[N,K])
        for i in range(N):
            for j in range(K):
                D[j] = np.linalg.norm(M[j,:] - S[i,:])
            R[i,np.argmin(D)] = 1
        #
        # Now we adjust the cluster centres. We first 
        # compute the matrix given by sum(R_ij x_i)
        #
        _M = np.matmul(R.T, S)
        #
        # Now we normalize the columns
        #
        col_sums = R.sum(axis=0)
        for j in range(K):
            if col_sums[j] != 0:
                M[j,:] = _M[j,:] / col_sums[j]
            else:
                M[j,:] = _M[j,:]
    return M, R
    

#############################################
# 
# The EM algorithm
#
#############################################
    

class GMM:
    
    #
    # Initialize the parameters. There are many ways to do
    # this, we could for instance start with some 
    # k-means step. We initialize randomly
    #
    def em_init(self, S, K):
        self.N = S.shape[0]
        #
        # Choose random cluster centres
        #
        self.means = S[np.random.randint(low=0, high=self.N, size=K), :]
        #
        # Initialize the parameters
        #
        self.K = K
        self.R = np.zeros(shape=[self.N,K])
        self.weights = np.ones(shape=[K],dtype=float) / K
        self.cov = []
        for _ in range(K):
            self.cov.append(np.eye(S.shape[1],dtype=float)) 
        
    #
    # Return computed responsibilities
    #
    def infer(self):
        return self.R

    
    #
    # Calculate loss function, i.e. the negative log likelihood
    #
    def loss(self, S):
        loss = 0
        for n in range(S.shape[0]):
            p = 0
            for k in range(self.K):
                p += self.weights[k] * multivariate_normal(
                    self.means[k],self.cov[k]).pdf(S[n])
            loss -= np.log(p)
        return loss
    
    #
    # E-step: determine responsibilities
    #
    def em_e_step(self, S):
        
        self.R = np.zeros(shape=(self.N,self.K))
        for k in range(self.K):
            self.R[:,k] = self.weights[k]*multivariate_normal(
                    self.means[k],self.cov[k]).pdf(S)
        #
        # And normalize 
        #
        self.R = (self.R.T / np.sum(self.R, axis=1)).T
        
    #
    # M-step
    #
    def em_m_step(self, S):
    
        Nk = np.sum(self.R, axis=0)    
        self.means = np.dot(self.R.T, S)
        #
        # Need to transpose as broadcasting
        # starts from last index
        #
        self.means = (self.means.T / Nk).T
    
        self.cov = []        
        for k in range(self.K):
            y = S[:,:] - self.means[k,:]
            #
            # Build an array O such that
            # O[n,:,:] is the outer product of y[n] 
            # with itself
            #
            outer = np.multiply.outer(y,y)
            O = np.diagonal(np.swapaxes(outer,1,2)).T
            #
            # Note that np.dot will sum along the second-to-last axis
            # of its second argument, so we need to apply swapaxes first
            #
            _cov = np.dot(self.R[:,k], np.swapaxes(O, 0,1)) / Nk[k]
            self.cov.append(_cov)
            
        self.weights = Nk / self.N
            
    
    ########################################################
    # Main EM algorithm. Fit model parameters for a
    # given sample S and return means and responsibilities
    # Parameter:
    #   S - the sample
    #   K - the nummber of clusters
    #   iterations - number of iterations to run
    #   verbose - print out some messages
    ######################################################## 
    def train(self, S, K=2, iterations=10, verbose=1):
        #
        # Initialize the cluster centres randomly
        #    
        self.em_init(S,K)
        #
        # Actual algorithm
        #
        for _ in range(iterations):
            #
            # E-step: calculate responsibilities
            #
            self.em_e_step(S)
    
            #
            # M-step: estimate parameters
            #        
            self.em_m_step(S)
                    
            if verbose and (0 == _ % 10):
                loss = self.loss(S)
                print("Completed step ",_, "loss is now: ", loss)
    

############################################
#
# Plot a sample set and cluster assignments
#
############################################

def plot_clusters(S, R, T, axis):
    for i in range(S.shape[0]):
        x = S[i,0]
        y = S[i,1]
        if R[i,0] >= 0.5 and T[i] == 0:
            axis.plot([x],[y],marker="o", color="red")
        elif R[i,0] < 0.5 and T[i] == 0:
            axis.plot([x],[y],marker="o", color="blue")
        elif R[i,0] >= 0.5 and T[i] == 1:
            axis.plot([x],[y],marker="d", color="red")
        elif R[i,0] < 0.5 and T[i] == 1:
            axis.plot([x],[y],marker="d", color="blue")
        else:
            raise ValueError("Unkown combination: R[i,0] = ", R[i,0])


############################################
# Load Iris data
############################################

def load_iris_data(offset, batch_size):

    df = pandas.read_csv('iris.data', 
                 header=None,sep=',',
                 names=["Sepal length", "Sepal width", "Petal length", "Petal width", "Species"])
    labels = df.loc[offset:offset + batch_size-1,"Species"].values
    labels[labels == 'Iris-setosa'] = 0
    labels[labels == 'Iris-versicolor'] = 1
    labels[labels == 'Iris-virginica'] = 2
    labels = labels.astype(dtype=int)
    
    target = np.eye(3)[labels]
    X = df.loc[offset:offset + batch_size-1, 
                  ["Sepal length","Sepal width", "Petal length", "Petal width"]].values
               
    return X, target


####################################################
# Parse arguments
####################################################
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set",
                    choices=["Iris", "Sample"],
                    default="Sample",
                    help="Data set to use"
                    )
    parser.add_argument("--save", 
                    type=int,
                    default=0,
                    help="Save generated images")
    args=parser.parse_args()
    return args

####################################################
# Utility function to print the results of the Iris 
# run
####################################################

def print_iris_cluster(R, T):
    results = np.zeros(shape=(3,3))
    for n in range(R.shape[0]):
        results[np.argmax(R[n]), np.argmax(T[n])] += 1


    table = pt.PrettyTable()
    flower_names = ['Cluster','Iris-setosa', 'Iris-versicolor','Iris-virginica']
    table.field_names = flower_names
    for _ in range(3):
        table.add_row([_,results[_,0], results[_,1], results[_,2]])
    print(table)

    #
    # Determine number of errors
    #
    for _ in range(3):
        assigned_flower = np.argmax(results[_,:])
        print("Assigned flower for cluster ",_," is ",flower_names[assigned_flower+1])
        results[_,assigned_flower] = 0
    errors = np.sum(results)
    print("Errors: ", errors)
    

############################################
#
# Main
#
############################################

args = get_args()

if args.data_set == "Sample":
    #
    # We use two clusters in 2-dimensional space
    #    
    d = 2
    K = 2
    size = 500
    steps = 100
    _M = np.array([[5,1], [1,4]])


    #
    # Prepare axis
    #
    fig = plt.figure(figsize=(12,10))
    ax_good_kmeans = fig.add_subplot(2,2,1)
    ax_bad_kmeans = fig.add_subplot(2,2,2)
    ax_good_em = fig.add_subplot(2,2,3)
    ax_bad_em = fig.add_subplot(2,2,4)



    S1, T1 = get_sample(_M,p=[0.5, 0.5], size=size, cov=[[0.5, 0.5], [0.5,0.5]])
    M1,R1 = kMeans(S1, K=2, iterations=steps)
    plot_clusters(S1,R1, T1, ax_good_kmeans)

    cov1 = np.asarray([[1.5, 0.0], [0.0,0.1]])
    cov2 = np.asarray([[0.8, 0.0], [0.0,0.8]])
    S2, T2 = get_sample(_M,p=[0.95, 0.05], size=size, cov=[cov1, cov2])
    M2,R2 = kMeans(S2, K=2, iterations=steps)
    plot_clusters(S2,R2, T2, ax_bad_kmeans)

    gmm = GMM()
    gmm.train(S1, K=2, iterations=steps, verbose=0)
    R3 = gmm.infer()
    plot_clusters(S1,R3, T1, ax_good_em)

    gmm.train(S2, K=2, iterations=steps, verbose=0)
    R4 = gmm.infer()
    plot_clusters(S2,R4,T2, ax_bad_em)


    if args.save == 1:
        fig.savefig("GMM.png")

    plt.show()

elif args.data_set == "Iris":
    #
    # Do the Iris data. This is a data
    # set with d=4, i.e. 4 features. Make sure that 
    # you have the Iris data file iris.data in your 
    # working directory
    #
    SI, TI = load_iris_data(0, 150)
    print("Running EM cluster analysis on Iris data set")
    gmm = GMM()
    gmm.train(SI, K=3, iterations=200, verbose=0)
    RI = gmm.infer()
    loss = gmm.loss(SI)
    print_iris_cluster(RI, TI)
    print("Final loss function: ", loss)

    print("Now running k-means")
    MI,RI = kMeans(SI, K=3, iterations=200)
    print_iris_cluster(RI, TI)
    
