#####################################################
#
# Some utility functions to test the TensorFlow
# implementation of the RBM
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
#
#####################################################

from __future__ import print_function  

import tensorflow as tf
import RBM.PCDTF

import numpy as np
from scipy.special import expit


#
# Floating point threshold
#
epsilon = 0.000001

#
# Test model building - this will make sure that
# everything is syntactically correct
#
def test_tc1():
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    rbm.prepare_training(1, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)


#
# Test one iteration of the training procedure
#
def test_tc2():
    #
    # Create machine
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    rbm.prepare_training(1, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # Now train the model
    #
    rbm.train(V, iterations=1, epochs=1)
    
    
#
# Test the positive phase model
#
def test_tc3():
    #
    # Create machine and build model
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    rbm.prepare_training(1, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # Run first step of the positive model
    #
    # all_ops = tf.get_default_graph().get_operations()
    E = rbm.run_operation("positive/E", feed_dict = {rbm.tf.S0 : V})
    #
    # Check it
    #
    _E = expit(rbm.beta*(np.matmul(V, rbm.W) + rbm.c))
    error = np.linalg.norm(E - _E)
    assert(error < epsilon)
    #
    # Now the next step 
    #
    _pos = np.tensordot(V, E, axes=((0),(0)))
    pos = rbm.run_operation("positive/pos", feed_dict = {rbm.tf.S0 : V})
    error = np.linalg.norm(pos - _pos)
    assert(error < epsilon)
    
    
#
# Test the negative phase model - forward pass
#
def test_tc4():
    #
    # Create machine and build model
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    rbm.prepare_training(1, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # First test that N is properly initialized
    #
    _N = rbm.session.run(rbm.tf.N)
    error = np.linalg.norm(_N - rbm.N)
    assert(error < epsilon)
    #
    # Now run the first operation - expectation value
    #
    E = rbm.run_operation("negative/E")
    #
    # Check it
    #
    _E = expit(rbm.beta*(np.matmul(rbm.N, rbm.W) + rbm.c))
    error = np.linalg.norm(E - _E)
    assert(error < epsilon)
    #
    # Now sample the hidden units and get the result plus
    # the used random values
    # 
    H_op = tf.get_default_graph().get_operation_by_name("negative/H")
    U_op = tf.get_default_graph().get_operation_by_name("negative/H_U")
    H, U = rbm.session.run([H_op.outputs[0], U_op.outputs[0]])
    _H = (U <= _E).astype(int)
    error = np.linalg.norm(_H - H)
    assert(error < epsilon)


#
# Test the negative phase model - backward pass
#
def test_tc5():
    #
    # Create machine and build model
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    rbm.prepare_training(1, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # First test that N is properly initialized
    #
    _N = rbm.session.run(rbm.tf.N)
    error = np.linalg.norm(_N - rbm.N)
    assert(error < epsilon)
    #
    # Now run the first operation - expectation value
    #
    E = rbm.run_operation("negative/E")
    #
    # Check it
    #
    _E = expit(rbm.beta*(np.matmul(rbm.N, rbm.W) + rbm.c))
    error = np.linalg.norm(E - _E)
    assert(error < epsilon)
    #
    # Now sample the hidden units and get the result plus
    # the used random values
    # 
    H_op = tf.get_default_graph().get_operation_by_name("negative/H")
    U_op = tf.get_default_graph().get_operation_by_name("negative/H_U")
    H, U = rbm.session.run([H_op.outputs[0], U_op.outputs[0]])
    _H = (U <= _E).astype(int)
    error = np.linalg.norm(_H - H)
    assert(error < epsilon)
    #
    # Next the backward pass
    # 
    b = rbm.run_operation("b")
    assert(np.linalg.norm(b - rbm.b) < epsilon)
    W = rbm.run_operation("W")
    assert(np.linalg.norm(W - rbm.W) < epsilon)
    _P = expit(rbm.beta*(np.matmul(H, np.transpose(rbm.W)) + rbm.b))
    P = rbm.run_operation("negative/P")
    error = np.linalg.norm(P - _P)
    #
    # It turns out that the implementations of expit and
    # tf.sigmoid give slightly different values, so let
    # us not be too picky here
    #
    assert(error < 0.02)
    Nb_op = tf.get_default_graph().get_operation_by_name("negative/Nb")
    U_op = tf.get_default_graph().get_operation_by_name("negative/Nb_U")
    Nb, U = rbm.session.run([Nb_op.outputs[0], U_op.outputs[0]])
    _Nb = (U <= _P).astype(int)
    error = np.linalg.norm(_Nb - Nb)
    assert(error < epsilon)


#
# Test the negative phase model - entire negative contribution
#
def test_tc6():
    #
    # Create machine and build model
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    rbm.prepare_training(1, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # First test that N is properly initialized
    #
    _N = rbm.session.run(rbm.tf.N)
    error = np.linalg.norm(_N - rbm.N)
    assert(error < epsilon)
    #
    # Get references to the two uniform random variables
    #
    Nb_U_op = tf.get_default_graph().get_operation_by_name("negative/Nb_U")
    H_U_op = tf.get_default_graph().get_operation_by_name("negative/H_U")
    #
    # Now run the negative phase - this should correspond to one
    # step. We also retrieve the values of the random variables
    #
    neg, H_U, Nb_U = rbm.session.run([rbm.tf.neg,H_U_op.outputs[0], Nb_U_op.outputs[0]])
    #
    # Now run one Gibbs sampling step manually, using the same
    # random values, and compare the results
    #
    _E = expit(rbm.beta*(np.matmul(rbm.N, rbm.W) + rbm.c))
    _H = (H_U <= _E).astype(int)
    _P = expit(rbm.beta*(np.matmul(_H, np.transpose(rbm.W)) + rbm.b))
    _Nb = (Nb_U <= _P).astype(int)
    _Eb = expit(rbm.beta*(np.matmul(_Nb, rbm.W) + rbm.c))
    _neg = np.tensordot(_Nb, _Eb, axes=((0),(0)))
    error = np.linalg.norm(_neg - neg)
    assert(error < epsilon)
    #
    # Verify that the negative particle states have been updated
    #
    N = rbm.session.run(rbm.tf.N)
    error = np.linalg.norm(_Nb - N)
    assert(error < epsilon)
    
#
# Test one entire training step
#
def test_tc7():
    #
    # Create machine and build model
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    weight_decay = 0.1
    rbm.prepare_training(1, weight_decay = weight_decay, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # First test that N is properly initialized
    #
    _N = rbm.session.run(rbm.tf.N)
    error = np.linalg.norm(_N - rbm.N)
    assert(error < epsilon)
    #
    # Get references to the two uniform random variables
    #
    Nb_U_op = tf.get_default_graph().get_operation_by_name("negative/Nb_U")
    H_U_op = tf.get_default_graph().get_operation_by_name("negative/H_U")
    #
    # Now run one training step. We also retrieve the values of the random variables
    # as well as pos and neg
    #
    _, ndw, H_U, Nb_U, neg, pos, db, dc, dW = rbm.session.run([rbm.tf.trainingStep,
                                            rbm.tf.norm_dW,
                                            H_U_op.outputs[0], 
                                            Nb_U_op.outputs[0],
                                            rbm.tf.neg, rbm.tf.pos,
                                            rbm.tf.db, rbm.tf.dc, rbm.tf.dW],
                                            feed_dict = {rbm.tf.S0 : V})    
    #
    # Make sure that the global step has been updated
    #
    global_step_v =  [v for v in tf.global_variables() if v.name == "globalStep:0"][0]
    global_step = rbm.session.run(global_step_v)
    assert(global_step == 1)
    #
    # Manually calculate positive phase and compare
    #
    _E = expit(rbm.beta*(np.matmul(V, rbm.W) + rbm.c))
    _pos = np.tensordot(V, _E, axes=((0),(0)))
    # print("_pos =", _pos)
    # print("pos = ", pos)
    assert(np.linalg.norm(_pos - pos) < 0.001)
    #
    # Next calculate new values of particle states and check
    # that they have been updated
    #
    _E = expit(rbm.beta*(np.matmul(rbm.N, rbm.W) + rbm.c))
    _H = (H_U <= _E).astype(int)
    _P = expit(rbm.beta*(np.matmul(_H, np.transpose(rbm.W)) + rbm.b))
    _Nb = (Nb_U <= _P).astype(int)
    N = rbm.session.run(rbm.tf.N)
    error = np.linalg.norm(_Nb - N)
    assert(error < epsilon)
    _Eb = expit(rbm.beta*(np.matmul(_Nb, rbm.W) + rbm.c))
    _neg = np.tensordot(_Nb, _Eb, axes=((0),(0)))
    assert(np.linalg.norm(_neg - neg) < 0.001)
    #
    # Test bias updates - batch size is one. Make sure
    # to use the new value of the particle state
    #
    _db = 0.05 *rbm.beta*np.sum(V - N, 0) 
    error = np.linalg.norm(_db - db)
    assert(error < 0.01)
    _E = expit(rbm.beta*(np.matmul(V, rbm.W) + rbm.c))
    _dc = 0.05*rbm.beta*np.sum(_E - _Eb, 0)
    error = np.linalg.norm(_dc - dc)
    assert(error < 0.001)
    #
    # Check weight updates
    #
    _dW = 0.05*(rbm.beta*(pos -neg)  - weight_decay*rbm.W)
    error = np.linalg.norm(_dW - dW)
    assert(error < 0.01)
    #
    # Finally make sure that the parameters have actually been updated
    #
    _W = rbm.W + _dW
    _b = rbm.b + _db
    _c = rbm.c + _dc
    W, b, c = rbm.session.run([rbm.tf.W, rbm.tf.b, rbm.tf.c])
    assert(np.linalg.norm(W - _W) < 0.01)
    
    
