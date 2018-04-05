#####################################
# Multinomial regression with Python
# 
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
#####################################

import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#
# A class to run multinomial regression 
#

class MultinomialRegression:

    
    def __init__(self, classes=3, input=4):
        # 
        # Initialize weights and bias
        #
        self.classes = classes
        self.inputs = 4
        self.W = np.random.randn(self.inputs+1, self.classes)    


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
        #
        # Finally we add an artificial column with all ones to 
        # model the bias
        self.X = np.concatenate((np.ones([batch_size, 1]), self.X), 1)
    
    def softmax(self, x):
        y = x
        for row in range(x.shape[0]):
            shiftx = x[row] - np.max(x[row])
            exps = np.exp(shiftx)
            y[row] = exps / np.sum(exps)
        return y

    def train(self, epochs, learning_rate = 0.01):
        #
        # Train the network with the loaded data
        #
        self.error_means = []
        self.error_norms = []
        for step in range(epochs):
            # 
            # First we compute the logits p_i. The activation is
            # given by the matrix product XW 
            #
            p = self.softmax(np.matmul(self.X, self.W))
            # 
            # Next we compute the error - this is the difference of
            # the expected targets and the probabilities
            #
            E = self.target - p
            #
            # finally we update the weights
            #
            self.W = self.W + learning_rate * np.matmul(np.transpose(self.X), E)
            #
            # Update statistics
            #
            self.error_means.append(np.mean(abs(E)))
            self.error_norms.append(np.linalg.norm(abs(E)))
        
        self.error_norms = np.sqrt(self.error_norms)
        return self.error_means, self.error_norms
    
    def infer(self):
        #
        # Inference on the loaded data
        #
        p = self.softmax(np.matmul(self.X, self.W))
        #
        # If p is greater than 0.5 we assign 1, else 0
        #
        results = np.heaviside(p - 0.5, 0)
        return results, self.target


    def visualize(self, epochs, save=False, 
                  filename="MultinomialRegression.png",
                  xlabel="Sepal length", 
                  ylabel="Sepal width",
                  zlabel="Petal length"):
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

        #
        # Next we plot all data points in a 2-dimensional area
        # 
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
        plt.show()        
    
#
# Main
#
        
        
#
# First we load test and training data from a file
#
MR = MultinomialRegression()
training_batch_size = 150
epochs = 500
MR.load_data(0,training_batch_size)
error_means, error_norms = MR.train(epochs, learning_rate=0.001)
results, labels = MR.infer()
errors = abs(results - labels)
error_count = 0
for i in range(training_batch_size):
    if errors[i].any():
        error_count = error_count + 1
print("Classification errors: ", error_count)
MR.visualize(epochs=epochs, save=False, filename="MultinomialRegression.png")

