#####################################################
#
# A restricted Boltzmann machine trained using the
# constrastive divergence algorithm
#
# see 
#
# G. Hinton, Training products of experts by 
# minimizing contrastive divergence, Journal Neural 
# Computation Vol. 14, No. 8 (2002), 1771--1800
#
# G.~Hinton,
# A practical guide to training restricted Boltzmann 
# machines, Technical Report University of Montreal 
# TR-2010-003 (2010)
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
#####################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import tempfile
import argparse


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
# A restricted Boltzmann machine, trained using the contrastive divergence
# algorithm
#        
class RBM:
    
    def __init__ (self, visible = 8, hidden = 3, beta = 1):
        self.visible = visible
        self.hidden = hidden
        self.beta = beta
        self.W = np.random.normal(loc = 0, scale = 0.01, size = (visible, hidden))
        self.b = np.zeros(shape = (1,visible))
        self.c = np.zeros(shape = (1,hidden))
        
    #
    # Run one step in a Gibbs sampling Markov chain. 
    # We sample the hidden units from the visible
    # units V and the visible units V' from the
    # hidden units. V' is returned
    # 
    def runGibbsStep(self, V, size = 1):
        #
        # Sample hidden units from visible units
        #      
        E = expit(self.beta*(np.matmul(V, self.W) + self.c))
        U = np.random.random_sample(size=(size, self.hidden))
        H = (U <= E).astype(int)
        #
        # and now sample visible units from hidden units
        #
        P = expit(self.beta*(np.matmul(H, np.transpose(self.W)) + self.b))
        U = np.random.random_sample(size=(size, self.visible))
        return (U <= P).astype(int), E

    #
    # Sample from the learned distribution, starting with a 
    # random value
    #
    def sample(self, iterations = 100, size = 1):
        return self.sampleFrom(np.random.randint(low=0, high=2, size=(size,self.visible)))

    #
    # Sample from the learned distribution, starting at some
    # initial value
    #
    def sampleFrom(self, initial, iterations = 100, size = 1):
        V = initial
        for _ in range(iterations):
            V, _ = self.runGibbsStep(V, size = size)
        return V
    #
    # Train the model on a training data mini batch
    # stored in V. Each row of V corresponds to one
    # sample. The number of columns of V should
    # be equal to the number of visible units
    #
    def train(self,  V, iterations = 100, step = 0.01):
        # 
        # Check geometry
        #
        batch_size = V.shape[0]
        if (V.shape[1] != self.visible):
            print("Shape of training data", V.shape)
            raise ValueError("Data does not match number of visible units")
        #
        # Prepare logs
        #
        dw = []
        error = []
        # 
        # Now do the actual training. First we calculate the expectation 
        # values of the hidden units given the visible units. The result
        # will be a matrix of shape (batch_size, hidden)
        # 
        for _ in range(iterations):
            #
            # Run one Gibbs sampling step and obtain new values
            # for visible units and previous expectation values
            #
            Vb, E = self.runGibbsStep(V, batch_size)
            # 
            # Calculate new expectation values
            #
            Eb = expit(self.beta*(np.matmul(Vb, self.W) + self.c))
            #
            # Calculate contributions of positive and negative phase
            # and update weights and bias
            #
            pos = np.tensordot(V, E, axes=((0),(0)))
            neg = np.tensordot(Vb, Eb, axes=((0),(0)))
            dW = step*self.beta*(pos -neg) / float(batch_size)
            self.W += dW
            self.b += step*self.beta*np.sum(V - Vb, 0) / float(batch_size)
            self.c += step*self.beta*np.sum(E - Eb, 0) / float(batch_size)
            #
            # Update logs
            #
            dw.append(np.linalg.norm(dW))
            recon_error =np.linalg.norm(V - Vb) 
            error.append(recon_error)
            if 0 == (_ % 2000):
                print("Iteration ",_," - reconstruction error is now", recon_error)
        return dw,error
    
    
    #
    # Visualize the weights
    #
    def showWeights(self, fig, cols, rows, x_pix, y_pix):
        for r in range(rows):
            for c in range(cols):
                j = r*cols + c
                #
                # We display the weigths connected to hidden unit j
                #
                w = self.W[:,j]
                #
                # Normalize
                #
                min = np.min(w)
                w = w + min
                max = np.max(w)
                w = w / max
                ax = fig.add_subplot(rows, cols, j+1)
                ax.imshow(w.reshape(x_pix, y_pix), "Greys")
                ax.set_yticks([],[])
                ax.set_xticks([],[])
    
        

####################################################
# Parse arguments
####################################################
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", 
                    type=int,
                    default=20,
                    help="Size of image")
    parser.add_argument("--patterns", 
                    type=int,
                    default=120,
                    help="Number of patterns that we store")
    parser.add_argument("--hidden", 
                    type=int,
                    default=128,
                    help="Number of hidden units")
    parser.add_argument("--beta", 
                    type=float,
                    default=10.0,
                    help="Inverse temperature")
    parser.add_argument("--step",
                    type=float,
                    default=0.01,
                    help="Step size of gradient descent")
    parser.add_argument("--train", 
                    type=int,
                    default=5000,
                    help="Number of iterations during training")
    parser.add_argument("--sample", 
                    type=int,
                    default=100000,
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
    args=parser.parse_args()
    return args


####################################################
#
# Main
#
####################################################

args = get_args()
#
# Create sample set
#
BAS = BAS(args.N)
V= BAS.getSample(size = args.patterns)
#
# Init RBM and train
# 
RBM = RBM(visible=args.N*args.N, hidden=args.hidden, beta = args.beta)
dw, error = RBM.train(V, iterations=args.train, step = args.step)
#
# Get file name
#
if args.save == 1:
    tmp = tempfile.mktemp(dir="/tmp")

#
# Now test inference
#
if args.run_reconstructions:
    tests = 5
    cols = 4
    fig = plt.figure(figsize=(5,cols*tests/5))
    for t in range(tests):
        # Determine pattern that we test
        u = np.random.randint(low = 0, high = args.patterns)
        I = V[u,:]
        # Plot original pattern
        ax = fig.add_subplot(tests,cols,cols*t+1)
        ax.set_yticks([],[])
        ax.set_xticks([],[])
        ax.imshow(I.reshape(args.N,args.N), "binary")
        sample = np.copy(I)
        # Flip some bits at random
        for i in range(args.errors):
            field = np.random.randint(0,args.N*args.N)
            if  I[field] == 0:
                sample[field] = 1
            else:
                sample[field] = 0
        # Sample
        print("Sampling reconstruction ", t)
        R0 = RBM.sampleFrom(initial = sample, iterations = int(args.sample / 2))
        R = RBM.sampleFrom(initial = R0, iterations = int(args.sample / 2))
        # Display distorted image
        ax = fig.add_subplot(tests,cols,cols*t+2)
        ax.set_yticks([],[])
        ax.set_xticks([],[])
        ax.imshow(sample.reshape(args.N,args.N), "binary")
        # Display reconstructions
        b0 = R0[0,:].reshape(args.N,args.N)    
        b = R[0,:].reshape(args.N,args.N)    
        ax = fig.add_subplot(tests,cols,cols*t+3)
        ax.set_yticks([],[])
        ax.set_xticks([],[])
        ax.imshow(b0, "binary")
        ax = fig.add_subplot(tests,cols,cols*t+4)
        ax.set_yticks([],[])
        ax.set_xticks([],[])
        ax.imshow(b, "binary")
    
    
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
    cols = 4
    rows = 4
    print("Sampling ", rows*cols, "images")
    V = RBM.sample(iterations = args.sample*100, size=rows*cols)
    fig = plt.figure(figsize=(5,5))
    for i in range(rows*cols):
        ax = fig.add_subplot(cols,rows,i+1)
        ax.set_yticks([],[])
        ax.set_xticks([],[])
        ax.imshow(V[i,:].reshape(args.N,args.N), "binary")

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
    f.close()
    print("Saved simulation description and results in ",outfile)


plt.show()



