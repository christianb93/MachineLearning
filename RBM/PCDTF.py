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
    def bernoulli(self, E, name):
        return tf.nn.relu(tf.sign(E - tf.random_uniform(tf.shape(E), 
                                                        dtype=tf.float64, 
                                                        minval=0, maxval=1.0, name = name + "_U")), 
                                                        name = name)
    
    
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
        self.sampling_prepared = 0
        
    #
    # Build the model that is responsible for the negative particles
    #
    def build_particle_model(self, batch_size, W, b, c):
        with tf.name_scope("negative") as scope:
            N = tf.get_variable(name="N",
                            dtype=tf.float64,
                            shape=[self.particles, self.visible],
                            initializer = tf.zeros_initializer())
            self.tf.N = N
            E = tf.sigmoid(self.beta*(tf.matmul(N, W) + c), 
                            name="E")
            H = self.bernoulli(E, name="H")
            P = tf.sigmoid(self.beta*(tf.matmul(H,W, transpose_b=True) + b),
                            name = "P")
            Nb = self.bernoulli(P, name="Nb")
            runStep = tf.assign(N, Nb)
            #
            # If Eb is invoked, this will trigger one sampling step
            #
            Eb = tf.sigmoid(self.beta*(tf.matmul(runStep, W) + c))
            self.tf.Eb = Eb
            self.tf.neg = tf.tensordot(runStep, Eb, axes=[[0],[0]], name="neg")
        return Eb, Nb, self.tf.neg
        
    #
    # Build the model that is calculating the positive phase
    #
    def build_input_model(self, batch_size, W, b, c):
        with tf.name_scope("positive") as scope:
            #
            # This is the expectation value of the hidden units
            #
            E = tf.sigmoid(self.beta*(tf.matmul(self.tf.S0, W) + c), name="E")
            #
            # and the positive phase
            #
            self.tf.pos = tf.tensordot(self.tf.S0, E, axes=[[0],[0]], name="pos")
        return E, self.tf.S0, self.tf.pos
        
        
    #
    # Build the tensorflow model. 
    #
    def build_model(self, batch_size, weight_decay, total_steps, initial_step_size):
        #
        # The parameters of the model. We use placeholders and copy the real values
        # into the model using tf.assign at run time
        # 
        self.tf = collections.namedtuple("tf", ['W0','b0','c0','N0', 'initSampler', 'W', 'c', 'b', 'trainingStep', 'assignStep', 'norm_dw', 'gibbsStep', 'P', 'initSample', 'reconError', 'db', 'dc', 'dW'])
        #
        # Global step
        #
        globalStep = tf.get_variable(name="globalStep", shape=[], initializer=tf.zeros_initializer())
        #
        # the learning rate
        #
        step = tf.cast(initial_step_size * (1.0 - (1.0*globalStep)/(1.0*total_steps)), dtype=tf.float64, name="step")
        #
        # The particles
        #
        self.tf.N0 = tf.placeholder(name="N0", dtype=tf.float64, shape=[self.particles, self.visible])
        #
        # The input batch
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
        Eb, Nb, neg = self.build_particle_model(batch_size, W, b, c)
        #
        # and the model for the positive phase
        #
        E, S, pos = self.build_input_model(batch_size, W, b, c)
        #
        # Bias and weight updates
        #
        dc = self.beta * step * tf.reduce_sum(E - Eb, 0) / float(batch_size) 
        db = self.beta * step * tf.reduce_sum(S - Nb, 0) / float(batch_size) 
        self.tf.db = db
        self.tf.dc = dc
        dW = step*(self.beta*(pos -neg) - weight_decay*W)  / float(batch_size)
        self.tf.dW = dW
        self.tf.norm_dW = tf.norm(dW)
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
        #
        # Next we build a model part which is used to measure the reconstruction error
        #
        P = tf.sigmoid(self.beta*(tf.matmul(self.tf.S0, W) + c))
        H = self.bernoulli(P, name="H")
        Sb = self.bernoulli(tf.sigmoid(
                self.beta*(tf.matmul(H,W, transpose_b=True) + b)), name="Sb")
        self.tf.recon_error = tf.norm(self.tf.S0 - Sb)
        

    def prepare_training(self, batch_size, weight_decay, total_steps, initial_step_size):
        #
        # Build TF model if not done yet
        #
        if 0 == self.prepared:
            #
            # Make sure that there is no old graph dangling around
            #
            tf.reset_default_graph()
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
            
        #
        # Prepare logs
        #
        dw = []
        errors = []
        
        session= self.prepare_training(batch_size, 
                                        weight_decay, 
                                        total_steps = iterations*epochs, 
                                        initial_step_size = step)
        
        for i in range(iterations):
            #
            # Update weights in model and get new values
            #
            _, ndw = session.run([self.tf.trainingStep, self.tf.norm_dW], 
                                feed_dict = {
                                                self.tf.S0 : V
                                            })
            dw.append(ndw)
            #
            # Compute reconstruction error every few iterations
            #
            if 0 == (self.global_step % 50):
                recon_error = self.session.run(self.tf.recon_error, 
                                                feed_dict = {
                                                            self.tf.S0 : V
                                                            })
                errors.append(recon_error)
                if 0 == (self.global_step % 500):
                    print("Iteration ",self.global_step,"recon error is ", recon_error)
            self.global_step +=1
        return dw, errors
    

    #
    # Clean up after training and copy parameters back
    # into class
    # 
    def postTraining(self):
        self.W, self.b, self.c = self.session.run([self.tf.W, self.tf.b, self.tf.c])
        self.prepared = 0
        self.session.close()
        tf.reset_default_graph()
        
    #
    # Tensorflow model for Gibbs sampling
    #
    def build_sampling_model(self, sample_size):
        #
        # The parameters of the model 
        # 
        self.tf = collections.namedtuple("tf", ['W0','b0','c0','V0','V', 'initSampler', 'runGibbsStep'])
        self.tf.V0 = tf.placeholder(name="V0", dtype=tf.int64, shape=[sample_size, self.visible])
        self.tf.W0 = tf.placeholder(name="W0", dtype=tf.float64, shape=[self.visible, self.hidden])
        self.tf.b0 = tf.placeholder(name="b0", dtype=tf.float64, shape=[1, self.visible])
        self.tf.c0 = tf.placeholder(name="c0", dtype=tf.float64, shape=[1, self.hidden])
        W = tf.get_variable(name="W", 
                            dtype=tf.float64, 
                            shape=[self.visible, self.hidden],
                            initializer = tf.zeros_initializer())
        b = tf.get_variable(name="b", 
                            dtype=tf.float64, 
                            shape=[1, self.visible],
                            initializer = tf.zeros_initializer())
        c = tf.get_variable(name="c", 
                            dtype=tf.float64, 
                            shape=[1, self.hidden],
                            initializer = tf.zeros_initializer())
        #
        # and the initial value of the visible units
        #
        self.tf.V0 = tf.placeholder(name="V0", dtype=tf.int64, shape=[sample_size, self.visible])
        # 
        # The state of the visible units
        #
        self.tf.V = tf.get_variable(name="V", 
                                    dtype=tf.float64, 
                                    shape=[sample_size, self.visible],
                                    initializer = tf.zeros_initializer())        
        #
        # and an operation to initialize them from the placeholders
        #
        self.tf.initSampler=tf.group(
                tf.assign(self.tf.V, tf.cast(self.tf.V0, dtype=tf.float64)),
                tf.assign(W, self.tf.W0),
                tf.assign(b, self.tf.b0),
                tf.assign(c, self.tf.c0)
                )
        #
        # Now we can build the actual model for the Gibbs sampling steps
        #
        E = tf.sigmoid(self.beta*(tf.matmul(self.tf.V, W) + c))
        H = self.bernoulli(E, name="H")
        Vb = self.bernoulli(tf.sigmoid(
                self.beta*(tf.matmul(H,W, transpose_b=True) + b)), name="Vb")
        self.tf.runGibbsStep=tf.assign(self.tf.V, Vb)

    #
    # Sample from the learned distribution, starting at some
    # initial value - use Tensorflow
    #
    def sampleFrom(self, initial, iterations = 100, size = 1):
        #
        # Reset default graph - we might be called with a different sample size
        # than during the previous call
        #
        tf.reset_default_graph()
        #
        # and build the model
        #
        sample_size = initial.shape[0]
        assert(size == sample_size)
        self.build_sampling_model(sample_size)
        #
        # Build a session and set V to their initial value
        # 
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        session.run([self.tf.initSampler], feed_dict = {
                self.tf.V0 : initial.astype(int),
                self.tf.W0 : self.W,
                self.tf.b0 : self.b,
                self.tf.c0 : self.c})
        for _ in range(iterations):
            session.run(self.tf.runGibbsStep)
            if 0 == (_ % 1000):
                print("Gibbs sampling step ",_," completed")
            
        result = session.run(self.tf.V)
        session.close()
        return result

    #
    # A helper function that will get an operation from the
    # current default graph and run it
    #
    def run_operation(self, name, feed_dict=None):
        #
        # Get a reference to the operation
        #
        op = tf.get_default_graph().get_operation_by_name(name)
        #
        # and run it
        #
        return self.session.run(op.outputs[0], feed_dict= feed_dict)
