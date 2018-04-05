############################################
# Multilayer network with backpropagation
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
############################################

from __future__ import print_function  
import pandas
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit


#
# A class to implement a multilayer network
#

class LayeredNetwork:

    
    def __init__(self, classes=3, input=4, hidden=[100,30]):
        self.classes = classes
        self.inputs = input
        self.hidden_layers = len(hidden)
        # 
        # Initialize weights 
        #
        self.W = []
        self.B = []
        hidden.append(self.classes)
        for i in range(self.hidden_layers):
            if (i == 0):
                self.W.append(np.random.randn(self.inputs, hidden[0]))
            else:
                self.W.append(np.random.randn(hidden[i-1], hidden[i]))
            self.B.append(np.zeros(hidden[i]))
        i = self.hidden_layers
        self.W.append(np.random.randn(hidden[i-1], self.classes))
        self.B.append(np.zeros(self.classes))
        


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
        # Extract first batch_size names (column 4) 
        # We use the loc method of a Pandas data frame to create an array
        # of labels, i.e. a matrix with one column and batch_size rows
        self.labels = df.loc[offset:offset + batch_size-1,"Species"].values
        #
        # Replace texts by numbers
        #
        self.labels[self.labels == 'Iris-setosa'] = 0
        self.labels[self.labels == 'Iris-versicolor'] = 1
        self.labels[self.labels == 'Iris-virginica'] = 2
        self.labels = self.labels.astype(dtype=int)
        #
        # Build a target vector with self.classes columns in 1-of-K encoding
        #
        self.target = np.eye(self.classes)[self.labels]

        #
        # Similarly, extract a matrix with the first batch_size samples 
        #
        self.X = df.loc[offset:offset + batch_size-1, 
                        ["Sepal length","Sepal width", "Petal length", "Petal width"]].values
    
    def softmax(self, x):
        y = x
        for row in range(x.shape[0]):
            shiftx = x[row] - np.max(x[row])
            exps = np.exp(shiftx)
            y[row] = exps / np.sum(exps)
        return y

    def infer(self):
        #
        # Inference on the loaded data
        #
        H = []
        for i in range(self.hidden_layers):
            if i == 0:
                H.append(expit(np.matmul(self.X, self.W[i]) + self.B[i]))
            else:
                H.append(expit(np.matmul(H[i-1], self.W[i]) + self.B[i]))
        O = self.softmax(np.matmul(H[self.hidden_layers-1], self.W[self.hidden_layers]) + 
                         self.B[self.hidden_layers])        
        #
        # If O is greater than 0.5 we assign 1, else 0
        #
        results = np.heaviside(O - 0.5, 0)
        return results, self.target,H,O



    def train(self, epochs, learning_rate = 0.01, beta = 0, early_stopping=0):
        #
        # Train the network with the loaded data
        #
        self.error_means = []
        self.error_norms = []
        self.weight_norms = []
        self.bias_norms = []
        for step in range(epochs):
            # 
            # First we compute the outputs of the layers one 
            # by one (forward propagation). This will give us the
            # output of the i-th hidden layer in H[i], starting with zero,
            # and the output of the output layer in O
            #
            results, target, H, O =  self.infer()
            errors = abs(results - self.target)
            # 
            # Compute number of errors
            #
            error_count = 0
            for i in range(training_batch_size):
                if errors[i].any():
                    error_count = error_count + 1

            #
            # Now do the actual backpropagation
            #
            weight_norms_step = 0
            bias_norms_step = 0
            error_norms_step = 0
            error_means_step = 0
            for i in range(self.hidden_layers, -1, -1):
                #
                # Determine errors
                #
                if i == self.hidden_layers:
                    E = target - O
                else:
                    E = H[i]*(1-H[i])*np.matmul(E, np.transpose(self.W[i+1]))
                #
                # and input of layer 
                #
                if i == 0:
                    input = np.transpose(self.X)
                else:
                    input = np.transpose(H[i-1])
                #
                # Now update weights and bias
                #
                if early_stopping == 0 or error_count > 0:
                    self.B[i] += learning_rate * np.sum(E, axis=0) - learning_rate*beta*self.B[i]
                    self.W[i] +=  learning_rate * np.matmul(input, E) - learning_rate*beta*self.W[i]
                #
                # and update some statistical information
                #
                error_norms_step += np.linalg.norm(E)
                error_means_step += np.mean(abs(E))
                weight_norms_step += np.linalg.norm(self.W[i])
                bias_norms_step += np.linalg.norm(self.B[i])
            #
            # update statistics after each step
            #
            self.weight_norms.append(weight_norms_step)
            self.bias_norms.append(bias_norms_step)
            self.error_norms.append(error_norms_step)
            self.error_means.append(error_means_step)
            if 0 == step % 100:
                print("In step ",step, "error norm is ",error_norms_step, "and we have",error_count,"errors")
            
            
        
        self.error_norms = np.sqrt(self.error_norms)
        return self.error_means, self.error_norms, error_count
    


    def visualize(self, epochs, save=True, show=True,
                  filename="MultilayerNetwork.png",
                  xlabel="Sepal length", 
                  ylabel="Sepal width",
                  zlabel="Petal length", 
                  show_weight_norms=0):
        #
        # First we visualize the rate of convergence by plotting the
        # mean and norm of the error vectors 
        #
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,2,1)
    
        ax.plot(range(epochs), self.error_means, label="Mean error")
        ax.plot(range(epochs), self.error_norms, label="Norm error")
        ax.legend(loc='upper right', 
                       shadow=False, 
                       fontsize='medium')
        ax.set(xlabel='Epoch',
          title='Training results')
        ax.grid()

        if show_weight_norms == 1:
            ax = fig.add_subplot(1,2,2)
            ax.plot(range(epochs), self.weight_norms, label="Weights")
            ax.plot(range(epochs), self.bias_norms, label="Bias")
            ax.legend(loc='upper right', 
                           shadow=False, 
                           fontsize='medium')
            ax.set(xlabel='Epoch',
                   title='Training results')
            ax.grid()

        #
        # Next we plot all data points in a 2-dimensional area
        # 
        if show_weight_norms == 0:
            series1 = self.X[:,1]
            series2 = self.X[:,2]
            series3 = self.X[:,3]
            colors = []
            for i in range(series1.shape[0]):
                if results[i,0] == 1:
                    colors.append('b')
                elif results[i,1] == 1:
                    colors.append('g')
                else:
                    colors.append('c')
                if (results[i] == labels[i]).all() == False:
                    colors[i] = 'r'
            ax = fig.add_subplot(1,2,2, projection = '3d')
            ax.scatter(series1, series2, series3, c=colors)
            ax.set(xlabel=xlabel,
                   ylabel=ylabel,
                   zlabel=zlabel,
                   title="Classification results")

        
        #
        # Display all graphs
        #
        fig.tight_layout()
        # fig.subplots_adjust(right=2.0, hspace=0.5)

        if save:
            plt.savefig(filename)
        if show:
            plt.show()        
    
#
# Main
#
        
# 
# Parse arguments
#
parser = argparse.ArgumentParser()
parser.add_argument("--steps", 
                    type=int,
                    default=500,
                    help="Number of iterations during training")
parser.add_argument("--hidden", 
                    type=int,
                    nargs="+",
                    default=[100,30],
                    help="Number of hidden units per layer as a list")
parser.add_argument("--learning_rate", 
                    type=float,
                    default=0.0005,
                    help="Learning rate")
parser.add_argument("--beta", 
                    type=float,
                    default=0.0,
                    help="Regularization parameter")
parser.add_argument("--early_stopping", 
                    type=int,
                    default=0,
                    help="Early stopping if the error rate is zero")
parser.add_argument("--display", 
                    type=int,
                    default=1,
                    help="Display results")
parser.add_argument("--save", 
                    type=int,
                    default=0,
                    help="Save results")
parser.add_argument("--show_weight_norms",
                    type=int,
                    default=0,
                    help="Visualize weight norms")
args = parser.parse_args()
epochs = args.steps
learning_rate = args.learning_rate
training_batch_size = 150


print("Building network with hidden layers", args.hidden)
print("Using ",epochs,"epochs for training with learning rate",learning_rate, "and regulator",args.beta)

#
# Load test and training data from a file
#
LN = LayeredNetwork(hidden=args.hidden)
LN.load_data(0,training_batch_size)
error_means, error_norms, error_count = LN.train(epochs, 
                                    learning_rate=learning_rate,
                                    beta = args.beta,
                                    early_stopping=args.early_stopping)
results, labels, _, _ = LN.infer()
errors = abs(results - labels)
error_count = 0
for i in range(training_batch_size):
    if errors[i].any():
        print("Sample ",i, "not correctly classified, features are ",LN.X[i])
        error_count = error_count + 1
print("Classification errors: ", error_count)
LN.visualize(epochs=epochs, 
             save=args.save, 
             show=args.display, 
             show_weight_norms=args.show_weight_norms,
             filename="LayeredNetwork.png")

