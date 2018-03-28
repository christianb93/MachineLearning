#####################################################
#
# Base class for restricted Boltzmann machines
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


class BaseRBM:
    
    
    #
    # Call this after the training has completed
    #
    def postTraining(self):
        pass
        
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
        E = expit(self.beta*(np.matmul(V.astype(int), self.W) + self.c), dtype=self.np_type)
        U = np.random.random_sample(size=(size, self.hidden)).astype(self.np_type)
        H = (U <= E).astype(int)
        #
        # and now sample visible units from hidden units
        #
        P = expit(self.beta*(np.matmul(H, np.transpose(self.W)) + self.b), dtype=self.np_type)
        U = np.random.random_sample(size=(size, self.visible)).astype(self.np_type)
        return (U <= P).astype(int), E

    #
    # Sample from the learned distribution, starting with a 
    # random value
    #
    def sample(self, iterations = 100, size = 1):
        return self.sampleFrom(np.random.randint(low=0, high=2, size=(size,self.visible)), iterations = iterations, size = size)

    #
    # Sample from the learned distribution, starting at some
    # initial value
    #
    def sampleFrom(self, initial, iterations = 100, size = 1):
        V = initial.astype(int)
        for i in range(iterations):
            V, _ = self.runGibbsStep(V, size = size)
            if (iterations > 1000):
                if 0 == i % 1000:
                    print("Sampling iteration ", i)
        return V
    
    
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
    
    #
    # Retrieve the weights and parameters
    #
    def getParameters(self):
        params = {}
        params['W'] = self.W
        params['b'] = self.b
        params['c'] = self.c
        return params
        
    #
    # Set parameter
    #
    def setParameters(self, params):
        self.W = params['W'].astype(self.np_type)
        self.b = params['b'].astype(self.np_type)
        self.c = params['c'].astype(self.np_type)

