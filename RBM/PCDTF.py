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
import tensorflow as tf
from scipy.special import expit
import collections
from . import Base


class PCDRBM (Base.BaseRBM):
    
    #
    # Utility function to build a submodel in order to sample
    # from a Bernoulli distribution
    #
    def bernoulli(self, E):
        return tf.nn.relu(tf.sign(E - tf.random_uniform(tf.shape(E), 
                                                        dtype=tf.float64, 
                                                        minval=0, maxval=1.0)))
    
    
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
        self.prepared = 0
        
    #
    # Build the model that is responsible for the negative particles
    #
    def build_particle_model(self, batch_size, W, b, c):
        N = tf.get_variable(name="N",
                            dtype=tf.float64,
                            shape=[self.particles, self.visible],
                            initializer = tf.zeros_initializer())
        self.tf.N = N
        E = tf.sigmoid(self.beta*(tf.matmul(N, W) + c))
        H = self.bernoulli(E)
        Nb = self.bernoulli(tf.sigmoid(
                self.beta*(tf.matmul(H,W, transpose_b=True) + b)))
        Eb = tf.sigmoid(self.beta*(tf.matmul(tf.assign(N, Nb), W) + c))
        self.tf.Eb = Eb
        self.tf.neg = tf.tensordot(N, Eb, axes=[[0],[0]])
        return Eb, N, self.tf.neg
        
    #
    # Build the model that is calculating the positive phase
    #
    def build_sample_model(self, batch_size, W, b, c):
        #
        # This is the expectation value of the hidden units
        #
        E = tf.sigmoid(self.beta*(tf.matmul(self.tf.S0, W) + c))
        #
        # and the positive phase
        #
        self.tf.pos = tf.tensordot(self.tf.S0, E, axes=[[0],[0]])
        return E, self.tf.S0, self.tf.pos
        

        
    #
    # Build the tensorflow model. 
    #
    def build_model(self, batch_size, weight_decay, total_steps, initial_step_size):
        #
        # The parameters of the model. We use placeholders and copy the real values
        # into the model using tf.assign at run time
        # 
        self.tf = collections.namedtuple("tf", ['W0','b0','c0','N0', 'initSampler', 'W', 'c', 'b', 'trainingStep', 'assignStep', 'norm_dw'])
        #
        # Global step
        #
        globalStep = tf.get_variable(name="globalStep", shape=[], initializer=tf.zeros_initializer())
        #
        # the learning rate
        #
        step = tf.cast(initial_step_size * (1.0 - (1.0*globalStep)/(1.0*total_steps)), dtype=tf.float64)
        #
        # The particles
        #
        self.tf.N0 = tf.placeholder(name="N0", dtype=tf.float64, shape=[self.particles, self.visible])
        #
        # The sample
        #
        self.tf.S0 = tf.placeholder(name="S0", dtype=tf.float64, shape=[batch_size, self.visible])
        #
        # The weights and bias vectors
        #
        self.tf.W0 = tf.placeholder(name="W0", dtype=tf.float64, shape=[self.visible, self.hidden])
        self.tf.b0 = tf.placeholder(name="b0", dtype=tf.float64, shape=[1, self.visible])
        self.tf.c0 = tf.placeholder(name="c0", dtype=tf.float64, shape=[1, self.hidden])
        #
        # Now define the corresponding variables
        #
        W = tf.get_variable(name="W", 
                            dtype=tf.float64, 
                            shape=[self.visible, self.hidden],
                            initializer = tf.zeros_initializer())
        self.tf.W = W
        b = tf.get_variable(name="b", 
                            dtype=tf.float64, 
                            shape=[1, self.visible],
                            initializer = tf.zeros_initializer())
        self.tf.b = b
        c = tf.get_variable(name="c", 
                            dtype=tf.float64, 
                            shape=[1, self.hidden],
                            initializer = tf.zeros_initializer())
        self.tf.c = c
        #
        # Now we build the model that is responsible for the particles
        #
        Eb, N, neg = self.build_particle_model(batch_size, W, b, c)
        #
        # and the model for the positive phase
        #
        E, S, pos = self.build_sample_model(batch_size, W, b, c)
        #
        # Bias and weight updates
        #
        dc = self.beta * step * tf.reduce_sum(E - Eb, 0) / float(batch_size) 
        db = self.beta * step * tf.reduce_sum(S - N, 0) / float(batch_size) 
        dW = step*self.beta*(pos -neg) / float(batch_size) - step*weight_decay*W / float(batch_size)
        self.norm_dW = tf.linalg.norm(dW)
        #
        # The training step. Running this operation will run one
        # training iteration
        #
        self.tf.trainingStep = tf.group(
                tf.assign(b, b + db),
                tf.assign(c, c + dc),
                tf.assign(W, W + dW),
                tf.assign(globalStep, globalStep+1)
        )
        #
        # We need an operation to initialize everything from the placeholders
        #
        self.tf.initSampler=tf.group(
                tf.assign(self.tf.N, self.tf.N0),
                tf.assign(W, self.tf.W0),
                tf.assign(b, self.tf.b0),
                tf.assign(c, self.tf.c0)
                )
        
        

    def prepare_training(self, batch_size, weight_decay, total_steps, initial_step_size):
        #
        # Build TF model if not done yet
        #
        if 0 == self.prepared:
            #
            # Build model
            #
            print("Building model")
            self.build_model(batch_size=batch_size, weight_decay = weight_decay, total_steps = total_steps, initial_step_size = initial_step_size)
            #
            # Create a session
            #
            session = tf.Session()
            session.run(tf.global_variables_initializer())
            #
            # and initialize weights and variables
            #
            session.run([self.tf.initSampler], feed_dict = {
                self.tf.N0 : self.N.astype(int),
                self.tf.W0 : self.W,
                self.tf.b0 : self.b,
                self.tf.c0 : self.c})
            self.session = session
            self.prepared = 1
        return self.session

        
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
        
        session= self.prepare_training(batch_size, weight_decay, iterations*epochs, step)
        
        for i in range(iterations):
            #
            # Update weights in model and get new values
            #
            _, self.W, self.b, self.c, ndw = session.run([self.tf.trainingStep, self.tf.W, self.tf.b, self.tf.c, self.norm_dW], feed_dict = {
            self.tf.S0 : V})
            dw.append(ndw)
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
    

