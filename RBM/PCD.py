#####################################################
#
# A restricted Boltzmann machine trained using the
# constrastive divergence algorithm
# 
# see:
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
from scipy.special import expit
from . import Base


class PCDRBM (Base.BaseRBM):
    
    def __init__(self, visible = 8, hidden = 3, particles = 10, beta=2.0):
        self.visible= visible
        self.hidden = hidden
        self.beta = beta
        self.particles = particles
        #
        # Initialize weights with a random normal distribution
        #
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(visible, hidden))
        #
        # set bias to zero
        # 
        self.b = np.zeros(dtype=float, shape=(1, visible))
        self.c = np.zeros(dtype=float, shape=(1, hidden))
        #
        # Initialize the particles
        #
        self.N = np.random.randint(low=0, high=2, size=(particles,self.visible))
        self.global_step = 0
        
    #
    # Train the model on a training data mini batch
    # stored in V. Each row of V corresponds to one
    # sample. The number of columns of V should
    # be equal to the number of visible units
    #
    def train(self, V, iterations, epochs, step=0.001, weight_decay=0.0001):
        # 
        # Check geometry
        #
        batch_size = V.shape[0]
        if (V.shape[1] != self.visible):
            print("Shape of training data", V.shape)
            raise ValueError("Data does not match number of visible units")
        initial_step_size = step
        #
        # Prepare logs
        #
        dw = []
        errors = []
        
        for i in range(iterations):
            #
            # Update step size - we do this linearly over time
            #
            step = initial_step_size * (1.0 - (1.0*self.global_step)/(1.0*iterations*epochs))
            #
            # First we compute the negative phase. We run the
            # Gibbs sampler for one step, starting at the previous state
            # of the particles self.N
            #
            self.N, _ = self.runGibbsStep(self.N, size=self.particles)
            #
            # and use this to calculate the negative phase
            # 
            Eb = expit(self.beta*(np.matmul(self.N, self.W) + self.c))
            neg = np.tensordot(self.N, Eb, axes=((0),(0)))
            #
            # Now we compute the positive phase. We need the
            # expectation values of the hidden units
            #
            E = expit(self.beta*(np.matmul(V, self.W) + self.c))
            pos = np.tensordot(V, E, axes=((0),(0)))
            #
            # Now update weights
            #
            dW = step*self.beta*(pos -neg) / float(batch_size) - step*weight_decay*self.W / float(batch_size)
            self.W += dW
            self.b += step*self.beta*np.sum(V - self.N, 0) / float(batch_size) 
            self.c += step*self.beta*np.sum(E - Eb, 0) / float(batch_size) 
            dw.append(np.linalg.norm(dW))
            #
            # Compute reconstruction error every few iterations
            #
            if 0 == (self.global_step % 50):
                Vb = self.sampleFrom(initial = V, size=batch_size, iterations = 1)
                recon_error = np.linalg.norm(V - Vb) 
                errors.append(recon_error)
                if 0 == (self.global_step % 500):
                    print("Iteration ",self.global_step,"recon error is ", recon_error)
            self.global_step +=1
        return dw, errors
    

