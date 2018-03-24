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


class CDRBM (Base.BaseRBM):
    
    def __init__ (self, visible = 8, hidden = 3, beta = 1):
        self.visible = visible
        self.hidden = hidden
        self.beta = beta
        self.W = np.random.normal(loc = 0, scale = 0.01, size = (visible, hidden))
        self.b = np.zeros(shape = (1,visible))
        self.c = np.zeros(shape = (1,hidden))
        self.global_step = 0
        
    #
    # Train the model on a training data mini batch
    # stored in V. Each row of V corresponds to one
    # sample. The number of columns of V should
    # be equal to the number of visible units
    #
    def train(self,  V, iterations = 100, epochs = 1, step = 0.01, weight_decay=0):
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
        errors = []
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
            errors.append(recon_error)
            if 0 == (self.global_step % 500):
                print("Iteration ", self.global_step," - reconstruction error is now", recon_error)
            self.global_step +=1 
        return dw,errors
    
    

