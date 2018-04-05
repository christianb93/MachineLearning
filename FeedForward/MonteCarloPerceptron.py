#####################################################
# Simple Monte Carlo Perceptron
#
# This is a simple implementation of
# a Baysesian perceptron, using
# a Metroplois-Hastings algorithm to
# sample from the posterior distribution of
# the weights to obtain the predictive distribution
# We use a Gaussian for both weights and bias as
# a prior distribution
#
# The algorithm proceeds in two phases. First, in the
# burn-in phase, we start a Metropolis-Hastings chain
# from the initial values of the weigths. In each
# step, we perform a Metropolis-Hastings update
# and calculate the acceptance probability from the
# energy difference between old and new state.
# After the burn-in phase, we assume that the chain
# is close to equilibrium and start to sample the
# value of the predictive distribution.
#
# see R.M. Neal,  Bayesian Learning for neural networks,
# Springer 1996 for the theoretical background
#
# Note that in this simple example, we use the same
# set of data for test and training (burn-in)
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
#####################################################

import argparse
import pandas
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

#
# A class to implement a simple Monte
# Carlo perceptron which uses MCMC to 
# create a sample in the space of weights 
#

class MonteCarloPerceptron:

    
    def __init__(self, features, var=1.0):
        #
        # Number of features
        #
        self.d = features
        #
        # Variance for weights
        #
        self.var = float(var)
        print(self.var)
        scale = np.sqrt(self.var, dtype=float)
        print("Standard deviation is ", scale)
        # 
        # Initialize weights and bias according to the prior
        # distribution, i.e. from a Gaussian with mean zero
        # and variance var
        #
        self.w = np.random.normal(size=self.d, loc=0, scale=scale)
        self.b = np.random.normal(loc=0, scale=scale)
        print("Initial weight:", self.w)
        print("Initial bias:  ", self.b)

    #
    # Calculate the loss function with respect to the loaded test data
    # (self.X, self.labels) and the given set of weights
    # The loss function is minus the logarithm of the likelihood
    # function 
    #
    def loss_function(self, w, b):
         p = expit(np.matmul(self.X,w) + b)
         #
         # We use numpy masked arrays to account for an overflow
         # because the argument to the log is zero
         #
         overflow = -1.0e+20
         l = - np.matmul(self.labels,  np.ma.log2(p).filled(overflow))
         l = l - np.matmul(1-self.labels, np.ma.log2(1-p).filled(overflow))
         return l

    # 
    # Calculate the energy function. The energy function is the sum
    # of a term coming from the prior distribution and the classical
    # loss function
    #
    def energy(self, w, b):
        return 1/(2*self.var)*(np.linalg.norm(w)*np.linalg.norm(w) + 
                  np.linalg.norm(b)*np.linalg.norm(b)) + self.loss_function(w, b)
    
    #
    # Load training data
    #
    def load_data(self,offset, batch_size):
        #
        # Read Iris data set. We use the pandas 
        # routine read_csv which returns a Pandas
        # data frame. A data frame is essentially a 
        # 2-dimensional array with column and row
        # labels
        df = pandas.read_csv('iris.data', 
                     header=None,sep=',',
                     names=["Sepal length", "Sepal width", "Petal length", "Petal width", "Species"])
        # 
        # Extract first batch_size names (column 4) and convert them to a number
        # We use the loc method of a Pandas data frame to create an array
        # of labels, i.e. a matrix with one column and 100 rows
        self.labels = df.loc[offset:offset + batch_size-1,"Species"].values
        #
        # Replace textual labels by 0 and 1
        #
        self.labels = np.where(self.labels == 'Iris-setosa', 0.0, 1.0)
        #
        # Similarly, extract a matrix with the first batch_size samples - we use two columns only
        #
        self.X = df.loc[offset:offset + batch_size-1, ["Sepal length","Petal length"]].values
        
        
    #
    # Propose a new value for the weights. We draw from a 
    # Gaussian proposal distribution
    #
    def propose(self):
        return self.w + np.random.normal(size=self.d), self.b + np.random.normal()
        
        
    #
    # Run the algorithm either in a burn-in phase (infer=False)
    # or in the sampling phase (infer=True). In the sample phase,
    # we also calculate the value of the predictive distribution
    # given the training data
    #
    # We return four arrays:
    # 
    # - the values of the energies for the weights in the chain
    #   (to support monitoring)
    # - the result vector 
    # - the labels (for comparison)
    # - a vector with the weights that the chain has visited
    #
    def run_chain(self,steps = 100, batch_size = 100, infer = False):
        energies = []
        weights = []
        accept = 0
        sample = np.zeros(batch_size)
        #
        # Now run the actual algorithm. In each steps, 
        # we draw new values and either accept or reject
        # them, based on the new energy
        #
        E = self.energy(self.w, self.b)
        for i in range(steps):
            w_new, b_new = self.propose()
            E_new = self.energy(w_new, b_new)
            if (np.random.random() <= np.exp(E - E_new)):
                self.w, self.b = w_new, b_new
                E = E_new
                accept += 1
            #
            # If we are in the inference phase, use current position for
            # sample
            # 
            if infer:
                sample += + expit(np.matmul(self.X,self.w) + self.b)
            # 
            # Append current energy to energy vector for 
            # monitoring purposes
            #
            energies.append(E)
            weights.append(self.w)
            
        print("Accepted ", accept," out of ", steps)
        return energies, np.heaviside(sample / steps - 0.5, 0), self.labels, weights
    
#####################################################
# Main
#####################################################
        

#
# Get arguments from command line
#
parser = argparse.ArgumentParser()
parser.add_argument("--burn_in",
                    type = int,
                   default=50,
                   help="Steps during burn-in phase")
parser.add_argument("--steps",
                    type=int,
                    default=5,
                    help="Steps during inference phase")
parser.add_argument("--show",
                    type=int,
                    default=1,
                    help="Display results")
parser.add_argument("--outfile",
                    default=None,
                    help="Filename for visualization output")
parser.add_argument("--var", 
                    default=1.0,
                    type=float,
                    help="Variance of prior distribution of weights")
args = parser.parse_args()


#
# First we load test and training data from a file
#
MCP = MonteCarloPerceptron(features = 2, var= args.var)
training_batch_size = 100
MCP.load_data(0,training_batch_size)


# 
# Start the algorithm with a burn-in phase
#
print("Using variance ",args.var, "for the prior weight distribution")
print("Starting burn-in phase, steps = ",args.burn_in)
energies, _, _, weights = MCP.run_chain(steps=args.burn_in)
#
# Now infer
#
print("Done, now starting inference phase, steps = ", args.steps)
_, results, labels, _ = MCP.run_chain(infer=True, steps=args.steps)
errors = abs(results - labels)
print("Classification errors: ", int(errors.sum()))

#
# Display results
#
fig = plt.figure(figsize=(10,8))
series1 = MCP.X[:,0]
series2 = MCP.X[:,1]
colors = np.where(results == 0, 100, 200)

ax = fig.add_subplot(2,1,2)
ax.set(title="Loss function during burn-in")
ax.plot(energies, label="Loss function during burn-in")

ax = fig.add_subplot(2,2,1)
ax.set(title="Classification results")
ax.scatter(series1, series2, c=colors, label="Classification results")

ax = fig.add_subplot(2,2,2)
ax.set(title="Weights")
series3=[]
series4=[]
for w in weights:
    series3.append(w[0])
    series4.append(w[1])
ax.plot(series3, series4, "--o")

if args.outfile:
    plt.savefig(args.outfile)

if args.show == 1:
    plt.show()

