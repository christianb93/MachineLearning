#####################################
# Logistic regression with Python
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
from scipy.special import expit
import matplotlib.pyplot as plt

#
# A class to run logistic regression 
#
class LogisticRegression:

    
    def __init__(self):
        # 
        # Initialize weights and bias
        #
        self.w = np.random.randn(2)
        self.b = np.random.randn()


    #
    # Calculate the loss function. We do not need this
    # for the actual algorithm but for monitoring convergence
    #
    def loss_function(self):
         p = expit(np.matmul(self.X,self.w) + self.b)
         #
         # We use numpy masked arrays to account for an overflow
         # because the argument to the log is zero
         overflow = -1.0e+20
         l = - np.matmul(self.labels,  np.ma.log2(p).filled(overflow))
         l = l - np.matmul(1-self.labels, np.ma.log2(1-p).filled(overflow))
         return l

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
        
        
    def create_data(self, batch_size, var=0.1):
        # 
        # We create two sets of sample data, assuming that the batch size
        # is an even number
        #
        cluster_size = int(batch_size / 2)
        set1 = np.random.multivariate_normal(mean = [6.5, 4.5],
                cov = [[var,0], [0,var]], 
                size = cluster_size)
        labels1 = np.zeros(cluster_size)
        labels1 = np.transpose(labels1)
        set2 = np.random.multivariate_normal(mean = [5, 1.5],
                cov = [[var,0], [0,var]], 
                size = cluster_size)
        labels2 = np.zeros(cluster_size) + 1 
        labels2 = np.transpose(labels2)
        self.X = np.concatenate((set1, set2))
        self.labels = np.concatenate((labels1, labels2))


    def train(self, epochs, learning_rate = 0.01):
        #
        # Train the network with the loaded data
        #
        self.error_means = []
        self.error_norms = []
        self.loss = []
        for step in range(epochs):
            # 
            # First we compute the logits p_i. The activation is
            # given by the matrix product X w^T (note that numpy will automatically
            # do the transpose)
            #
            p = expit(np.matmul(self.X,self.w) + self.b)
            # 
            # Next we compute the error - this is the difference of
            # the expected labels and the probabilities
            #
            error = self.labels - p
            #
            # finally we update the weights
            #
            self.w = self.w + learning_rate * np.matmul(error,self.X)
            self.b = self.b + learning_rate * np.sum(error)
            #
            # Update statistics
            #
            self.error_means.append(np.mean(abs(error)))
            self.error_norms.append(np.linalg.norm(abs(error)))
            self.loss.append(self.loss_function())
            
        
        self.error_norms = np.sqrt(self.error_norms)
        return self.error_means, self.error_norms
    
    def infer(self):
        #
        # Inference on the loaded data
        #
        p = expit(np.matmul(self.X, self.w))
        #
        # If p is greater than 0.5 we assign 1, else 0
        #
        results = np.heaviside(p - 0.5, 0)
        return results, self.labels


    def visualize(self, epochs, save=True, 
                  filename="LogisticRegression.png",
                  xlabel="Sepal length", 
                  ylabel="Petal length"):
        #
        # First we visualize the rate of convergence by plotting the
        # mean and norm of the error vectors 
        #
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].plot(range(epochs), self.error_means, label="Mean error")
        ax[0].plot(range(epochs), self.error_norms, label="Norm error")
        ax[0].plot(range(epochs), self.loss / np.max(self.loss), label="Loss function")
        ax[0].legend(loc='upper right', 
                       shadow=False, 
                       fontsize='medium')
        ax[0].set(xlabel='Epoch',
          title='Training results')
        ax[0].grid()

        #
        # Next we plot all data points in a 2-dimensional area
        # 
        series1 = self.X[:,0]
        series2 = self.X[:,1]
        colors = np.where(results == 0, 100, 200)
        ax[1].scatter(series1, series2, c=colors)
        ax[1].set(xlabel=xlabel,
              ylabel=ylabel, 
              title="Classification results")

        #
        # Display all graphs
        #
        fig.tight_layout()
        if save:
            plt.savefig(filename, dpi=200)
        plt.show()        
    
#
# Main
#
        
        
#
# First we load test and training data from a file
#
LogisticRegression = LogisticRegression()
training_batch_size = 100
epochs = 15
LogisticRegression.load_data(0,training_batch_size)
error_means, error_norms = LogisticRegression.train(epochs)
results, labels = LogisticRegression.infer()
errors = results - labels
print("Classification errors: ", int(abs(errors.sum())))
LogisticRegression.visualize(epochs=epochs, save=False, 
                            filename="LogisticRegression.png")

#
# Now we create an artifical sample set 
#
LogisticRegression.create_data(batch_size=500, var=.7)
#
# and train and evaluate
#
epochs = 150
error_means, error_norms = LogisticRegression.train(epochs=epochs)
results, labels = LogisticRegression.infer()
errors = abs(results - labels)
error_count = 0
for i in range(training_batch_size):
    if errors[i].any():
        error_count = error_count + 1
LogisticRegression.visualize(epochs=epochs, save=False, 
                             filename="LogisticRegression.png",
                             xlabel='x',
                             ylabel='y')

