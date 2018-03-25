#####################################################
#
# Train and test a restricted Boltzmann machine
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

from __future__ import print_function  

import RBM.CD
import RBM.PCD



import numpy as np
import matplotlib.pyplot as plt
import tempfile
import argparse
import time
import datetime
from sklearn.datasets import fetch_mldata



#
# Utility class to generate a pattern from the bars-and-stripes
# pattern set (MacKay, Information Theory, Inference and learning
# algorithms, section 43)
# 
class BAS:
    
    def __init__ (self, N = 4):
        self.N = N
    
    def createMatrix(self, orientation = 0, number = 0):
        #
        # Create a 4x4 matrix out of the bars-and-stripes
        # collection
        #
        values = np.zeros((self.N,1))
        for i in range(self.N):
            values[i] = (number >> i) % 2
        if (orientation == 0):
            return np.matmul(values, np.ones((1,self.N)))
        else:
            return np.transpose(np.matmul(values, np.ones((1,self.N))))
        
    def createVector(self, orientation = 0, number = 0):
        M = self.createMatrix(orientation = orientation, number = number)
        return M.reshape((self.N*self.N,1))
    
    #
    # Return a matrix with a given number of 
    # samples. The result will be stored in a
    # matrix with shape (size, N*N)
    #
    def getSample(self, size = 30):
        if size > 2*(2**self.N) - 2:
            raise ValueError("Cannot generate that many samples")
        if 0 != (size % 2):
            raise ValueError("Number of samples must be even")
        images = []        
        for n in range(int(size / 2)):
            a = self.createVector(1,n+1)
            images.append(a)
            b = self.createVector(0,n+1)
            images.append(b)
        V = np.concatenate(images, axis=1)
        return np.transpose(V)

        
#
# A utility class to manage training data sets
# that consist of quadratic images
#
class TrainingData: 
    
    def __init__(self, N = 6, ds_size = 80, ds = "BAS"):
        self.ds = ds
        if ds == "BAS":
            self.BAS = BAS(args.N)
            self.ds_size = ds_size
            self.S = self.BAS.getSample(size = ds_size)
        elif ds == "MNIST":
            if (N != 28):
                raise ValueError("Please use N = 28 for the MNIST data set")
            mnist = fetch_mldata("MNIST original")
            label = mnist['target']
            mnist = mnist.data
            mnist = ((mnist / 255.0) + 0.5).astype(int)
            images = []
            for i in range(ds_size):
                digit = i % 10
                u = np.where(label == digit)[0]
                images.append(mnist[u[i // 10], None,:])
            self.S = np.concatenate(images, axis=0)
            self.ds_size = ds_size
        else:
            raise ValueError("Unknown data set name")
            
    def get_batch(self, batch_size = 10):
        images = []        
        for i in range(batch_size):
            u = np.random.randint(low = 0, high = self.ds_size)
            images.append(self.S[u,None,:])
        return np.concatenate(images, axis=0)


####################################################
# Parse arguments
####################################################
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", 
                    type=int,
                    default=6,
                    help="Size of image")
    parser.add_argument("--patterns", 
                    type=int,
                    default=20,
                    help="Number of patterns that we store")
    parser.add_argument("--hidden", 
                    type=int,
                    default=16,
                    help="Number of hidden units")
    parser.add_argument("--beta", 
                    type=float,
                    default=2.0,
                    help="Inverse temperature")
    parser.add_argument("--step",
                    type=float,
                    default=0.05,
                    help="Step size of gradient descent")
    parser.add_argument("--iterations", 
                    type=int,
                    default=1,
                    help="Number of iterations per batch during training")
    parser.add_argument("--epochs", 
                    type=int,
                    default=30000,
                    help="Number of epochs (batches) during training")
    parser.add_argument("--sample", 
                    type=int,
                    default=100,
                    help="Number of iterations during sampling")
    parser.add_argument("--errors", 
                    type=int,
                    default=3,
                    help="Number of points in the sample pattern that we flip")
    parser.add_argument("--save", 
                    type=int,
                    default=0,
                    help="Save results")
    parser.add_argument("--run_samples", 
                    type=int,
                    default=0,
                    help="Run test samples from full model")
    parser.add_argument("--run_reconstructions", 
                    type=int,
                    default=0,
                    help="Run reconstruction tests")
    parser.add_argument("--show_weights", 
                    type=int,
                    default=0,
                    help="Display weights after training")
    parser.add_argument("--show_metrics", 
                    type=int,
                    default=0,
                    help="Display a few metrics after training")
    parser.add_argument("--algorithm", 
                    choices=["CD", "PCD"],
                    default="CD",
                    help="Algorithm: contrastive divergence (CD) or PCD")
    parser.add_argument("--batch_size", 
                    type=int,
                    default=10,
                    help="Batch size")                    
    parser.add_argument("--weight_decay",
                    type=float,
                    default=0.0001,
                    help="Weight decay")                    
    parser.add_argument("--data",
                    choices=["BAS", "MNIST"],
                    default="BAS",
                    help="Data set")                    
    args=parser.parse_args()
    return args


#
# Utility function to properly display an array
# as an N x N binary image
#
def show_pattern(ax, v):
    ax.set_yticks([],[])
    ax.set_xticks([],[])
    ax.imshow(v.reshape(args.N,args.N), "binary")
    

####################################################
#
# Main
#
####################################################

args = get_args()
#
# Create sample set
#
TrainingData = TrainingData(N = args.N, 
                            ds_size = args.patterns, 
                            ds = args.data)

#
# Init RBM and train
# 
dw = []
error = []

if args.algorithm == "CD":
    RBM = RBM.CD.CDRBM(visible=args.N*args.N, hidden=args.hidden, beta = args.beta)
elif args.algorithm == "PCD":
    RBM = RBM.PCD.PCDRBM(visible=args.N*args.N, hidden=args.hidden, beta = args.beta, particles=args.batch_size)
else:
    raise ValueError("Unknown algorithm")
start_time = time.time()
print("Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


for e in range(args.epochs):
    V = TrainingData.get_batch(batch_size = args.batch_size)
    _dw, _error = RBM.train(V, 
                          iterations=args.iterations, 
                          epochs = args.epochs,
                          step = args.step, 
                          weight_decay=args.weight_decay)
    dw.append(_dw)
    if len(_error) > 0:
        error.append(_error)

end_time = time.time()
run_time = end_time - start_time

#
# Get file name
#
if args.save == 1:
    tmp = tempfile.mktemp(dir="/tmp")

#
# Now test reconstructions
#
if args.run_reconstructions:
    tests = 8
    cols = 4
    fig = plt.figure(figsize=(5,cols*tests/5))
    #
    # Determine a sample set that we use for testing
    #
    I =  TrainingData.get_batch(batch_size = tests)
    #
    # Now plot the original patterns
    #
    for t in range(tests):
        show_pattern(fig.add_subplot(tests,cols,cols*t+1), I[t,:])

    # 
    # Flip some bits at random in each
    # of the rows
    # 
    sample = np.copy(I)
    for t in range(tests):
        for i in range(args.errors):
            field = np.random.randint(0,args.N*args.N)
            sample[t,field] = (1 if I[t,field] == 0 else 0)

    # 
    # Sample
    #
    print("Sampling reconstructions")
    R0 = RBM.sampleFrom(initial = sample, iterations = int(args.sample / 2))
    R = RBM.sampleFrom(initial = R0, iterations = int(args.sample / 2))

    #
    # Display results
    #
    for t in range(tests):
        # Display distorted image
        show_pattern(fig.add_subplot(tests,cols,cols*t+2), sample[t,:])
        # Display reconstructions
        show_pattern(fig.add_subplot(tests,cols,cols*t+3), R0[t,:])        
        show_pattern(fig.add_subplot(tests,cols,cols*t+4), R[t,:])
    
    
    if args.save == 1:
        outfile = tmp + "_RBMPartI.png"
        print("Saving simulation results part I to ",outfile)
        fig.savefig(outfile)

#
# Display metrics
#
if args.show_metrics == 1:

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Change of weights")
    ax.plot(dw)

    ax = fig.add_subplot(1,2,2)
    ax.plot(error,"y")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reconstruction error")

    if args.save == 1:
        outfile = tmp + "_RBMPartII.png"
        print("Saving simulation results part II to ",outfile)
        fig.savefig(outfile)
    
    
#
# Now sample a few images and display them
#
if args.run_samples == 1:
    cols = 5
    rows = 5
    sampling_start_time = time.time()
    print("Sampling ", rows*cols, "images")
    print("Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    V = RBM.sample(iterations = args.sample, size=rows*cols)
    fig = plt.figure(figsize=(5,5))
    for i in range(rows*cols):
        show_pattern(fig.add_subplot(cols,rows,i+1), V[i,:])
    end_time = time.time()
    run_time_sampling = end_time - sampling_start_time
    

    if args.save == 1:
        outfile = tmp + "_RBMPartIII.png"
        print("Saving simulation results part III to ",outfile)
        fig.savefig(outfile)


#
# Display weights
#
if args.show_weights == 1:
    fig = plt.figure(figsize=(10,10))
    RBM.showWeights(fig, 4, 4, args.N, args.N)
    if args.save == 1:
        outfile = tmp + "_RBMPartIV.png"
        print("Saving simulation results part IV to ",outfile)
        fig.savefig(outfile)

if args.save == 1:
    outfile = tmp + "_RBMDesc.txt"
    f= open(outfile, "w")
    print(args, file=f)
    print("Run time: ", str(datetime.timedelta(seconds=int(run_time))), file = f)
    if args.run_samples == 1:
        print("Run time sampling: ", str(datetime.timedelta(seconds=int(run_time_sampling))), file=f)    
    f.close()
    print("Saved simulation description and results in ",outfile)


print("Run time: ", str(datetime.timedelta(seconds=int(run_time))))
if args.run_samples == 1:
    print("Run time sampling: ", str(datetime.timedelta(seconds=int(run_time_sampling))))    


plt.show()



