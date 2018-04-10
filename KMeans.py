#############################################
#
# K-means algorithm
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
#
#############################################


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

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
# and standard deviations
# 
# The centre points are assumed to be given
# as a matrix with K rows. The vector p is 
# a vector with non-negative elements that
# sum up to one 
# 
# Returns the sample and the true cluster
#############################################

def get_sample(M, p, stddev, size=10):
    #
    # Number of clusters
    #
    X = []
    T = []
    for i in range(size):
        #
        # First determine the cluster by drawing
        # from the distribution given by p
        #
        k = draw(p) - 1
        T.append(k)
        #
        # Then draw from the normal distribution with mean M[k,:]
        #
        _X = np.random.normal(loc=M[k,:], scale=stddev[k], size=[1, M.shape[1]])
        X.append(_X)
    return np.concatenate(X), T
    
    

#############################################
# 
# The K-means algorithm
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
    

############################################
#
# Plot a sample set and cluster assignments
#
############################################

def plot_clusters(S, R, T, axis):
    for i in range(S.shape[0]):
        x = S[i,0]
        y = S[i,1]
        if R[i,0] == 1 and T[i] == 0:
            axis.plot([x],[y],marker="o", color="red")
        elif R[i,0] == 0 and T[i] == 0:
            axis.plot([x],[y],marker="o", color="blue")
        elif R[i,0] == 1 and T[i] == 1:
            axis.plot([x],[y],marker="d", color="red")
        else:
            axis.plot([x],[y],marker="d", color="blue")

############################################
#
# Main
#
############################################



#
# We use two clusters in 2-dimensional space
#    
d = 2
K = 2
size = 100
_M = np.array([[5,1], [1,4]])


#
# Prepare axis
#
fig = plt.figure(figsize=(12,5))
ax_good = fig.add_subplot(1,2,1)
ax_bad = fig.add_subplot(1,2,2)


print("Drawing sample one")
S, T = get_sample(_M,p=[0.5, 0.5], size=size, stddev=[0.6, 0.6])

print("Now starting k-Means")
M,R = kMeans(S, K=2, iterations=100)

#
# Plot results
#
plot_clusters(S,R, T, ax_good)

#
# Repeat the same for a "bad" example
#
print("Drawing sample two")
S, T = get_sample(_M,p=[0.95, 0.05], size=size, stddev=[0.8, 0.5])
print("Now starting k-Means")
M,R = kMeans(S, K=2, iterations=10)
plot_clusters(S,R, T, ax_bad)


#fig.savefig("KMeans.png")

plt.show()
