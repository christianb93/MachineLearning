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
import RBM.PCD
import RBM.CD

import numpy as np
from scipy.special import expit


#
# Floating point threshold
#
epsilon = 0.000001

#
# A utility function to get a reference to a variable
# in the default graph
#
def get_variable_by_name(name):
    return  [v for v in tf.global_variables() if v.name == name]
    
    
#
# A helper function that will get an operation from the
# current default graph and run it
#
def run_operation(rbm, name, feed_dict=None):
    #
    # Get a reference to the operation
    #
    op = tf.get_default_graph().get_operation_by_name(name)
    #
    # and run it
    #
    return rbm.session.run(op.outputs[0], feed_dict= feed_dict)
    
#
# Test model building - this will make sure that
# everything is syntactically correct
#
def test_tc1():
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    rbm.prepare_training(1, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Check that the TensorFlow model parameters have been initialized
    #
    N, W, b, c = rbm.session.run([rbm.tf.N, rbm.tf.W, rbm.tf.b, rbm.tf.c])
    error = np.linalg.norm(W - rbm.W)
    assert(error < epsilon)
    error = np.linalg.norm(N - rbm.N)
    assert(error < epsilon)


#
# Test one iteration of the training procedure
#
def test_tc2():
    #
    # Create machine
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    rbm.prepare_training(1, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    tf.summary.FileWriter('/tmp/logs', rbm.session.graph)
    
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
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=4)
    rbm.prepare_training(batch_size=4, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(4,3))
    V[0,1] = 1
    V[1,2] = 1
    V[2,0] = 1
    V[3,1] = 1
    #
    # Run first step of the positive model
    #
    # all_ops = tf.get_default_graph().get_operations()
    E = run_operation(rbm, "positive/E", feed_dict = {rbm.tf.S0 : V})
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
    pos = run_operation(rbm, "positive/pos", feed_dict = {rbm.tf.S0 : V})
    error = np.linalg.norm(pos - _pos)
    assert(error < epsilon)
    
    
#
# Test the negative phase model - forward pass
#
def test_tc4():
    #
    # Create machine and build model
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=4)
    rbm.prepare_training(batch_size=4, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(4,3))
    V[0,1] = 1
    V[1,2] = 1
    V[2,0] = 1
    V[3,1] = 1
    #
    # First test that N is properly initialized
    #
    _N = rbm.session.run(rbm.tf.N)
    error = np.linalg.norm(_N - rbm.N)
    assert(error < epsilon)
    #
    # Now run the first operation - expectation value
    #
    E = run_operation(rbm, "negative/E")
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
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=4)
    rbm.prepare_training(batch_size=4, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(4,3))
    V[0,1] = 1
    V[1,2] = 1
    V[2,0] = 1
    V[3,1] = 1
    #
    # First test that N is properly initialized
    #
    _N = rbm.session.run(rbm.tf.N)
    error = np.linalg.norm(_N - rbm.N)
    assert(error < epsilon)
    #
    # Now run the first operation - expectation value
    #
    E = run_operation(rbm, "negative/E")
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
    b = run_operation(rbm, "b")
    assert(np.linalg.norm(b - rbm.b) < epsilon)
    W = run_operation(rbm, "W")
    assert(np.linalg.norm(W - rbm.W) < epsilon)
    _P = expit(rbm.beta*(np.matmul(H, np.transpose(rbm.W)) + rbm.b))
    P = run_operation(rbm, "negative/P")
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
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=4)
    rbm.prepare_training(batch_size=4, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(4,3))
    V[0,1] = 1
    V[1,2] = 1
    V[2,0] = 1
    V[3,1] = 1
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
    # Note that this will NOT yet update N, only Nb
    #
    neg, H_U, Nb_U,  = rbm.session.run([rbm.tf.neg,H_U_op.outputs[0], Nb_U_op.outputs[0]])
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
# Test one entire training step
#
def test_tc7():
    #
    # Create machine and build model
    #
    weight_decay = 0.001
    batch_size = 4
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=4)
    rbm.prepare_training(batch_size=4, weight_decay = weight_decay, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(4,3))
    V[0,1] = 1
    V[1,2] = 1
    V[2,0] = 1
    V[3,1] = 1
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
    db_op = tf.get_default_graph().get_operation_by_name("delta/db")
    dc_op = tf.get_default_graph().get_operation_by_name("delta/dc")
    dW_op = tf.get_default_graph().get_operation_by_name("delta/dW")
    _, ndw, H_U, Nb_U, neg, pos, db, dc, dW = rbm.session.run([rbm.tf.trainingStep,
                                            rbm.tf.norm_dW,
                                            H_U_op.outputs[0], 
                                            Nb_U_op.outputs[0],
                                            rbm.tf.neg, rbm.tf.pos,
                                            db_op.outputs[0], 
                                            dc_op.outputs[0], 
                                            dW_op.outputs[0]],
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
    _db = 0.05 *rbm.beta*np.sum(V - N, 0) / float(batch_size)
    error = np.linalg.norm(_db - db)
    assert(error < 0.01)
    _E = expit(rbm.beta*(np.matmul(V, rbm.W) + rbm.c))
    _dc = 0.05*rbm.beta*np.sum(_E - _Eb, 0) / float(batch_size)
    error = np.linalg.norm(_dc - dc)
    assert(error < 0.001)
    #
    # Check weight updates
    #
    _dW = 0.05*(rbm.beta*(pos -neg)  - weight_decay*rbm.W) / float(batch_size)
    error = np.linalg.norm(_dW - dW)
    assert(error < 0.01)
    #
    # Finally make sure that the parameters have actually been updated
    #
    _W = rbm.W + _dW
    _b = rbm.b + _db
    _c = rbm.c + _dc
    W, b, c = rbm.session.run([rbm.tf.W, rbm.tf.b, rbm.tf.c])
    assert(np.linalg.norm(W - _W) < 0.008)
    
#
# Test sampling
#
def test_tc8():
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
    for _ in range(10):
        rbm.train(V, iterations=1, epochs=10)
    #
    # and sample
    # 
    V = rbm.sample(iterations = 5, size=4)
    assert(V.shape[0] == 4)


#
# Test one sampling step in detail
#
def test_tc9():
    #
    # Create machine
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    rbm.prepare_training(batch_size = 2, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    # Prepare input data
    #
    V = np.zeros(shape=(2,3))
    V[0,1] = 1
    #
    # Now train the model
    #
    for _ in range(10):
        rbm.train(V, iterations=1, epochs=10)
    #
    # Get the parameters
    #
    rbm.postTraining()
    params = rbm.getParameters()
    _W = params['W']
    _b = params['b']
    _c = params['c']
    initial = np.zeros(shape=(2,3))
    initial[0,1] = 1
    #
    # Build sampling model
    #
    tf.reset_default_graph()
    rbm.build_sampling_model(sample_size=2)
    #
    # Prepare session
    #
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    rbm.tf.V.load(initial, session)
    rbm.tf.W.load(_W, session)
    rbm.tf.b.load(_b, session)
    rbm.tf.c.load(_c, session)
    #
    # Get references to the two random variables used for sampling
    #
    Vb_U_op = tf.get_default_graph().get_operation_by_name("sampling/Vb_U")
    H_U_op = tf.get_default_graph().get_operation_by_name("sampling/H_U")
    _, Vb_U, H_U = session.run([rbm.tf.runGibbsStep, 
                                Vb_U_op.outputs[0], 
                                H_U_op.outputs[0]])
    #
    # and get result of sampling
    #
    V = session.run(rbm.tf.V)
    #
    # Now sample one step manually and compare
    #
    _E = expit(rbm.beta*(np.matmul(initial, _W) + _c))
    _H = (H_U <= _E).astype(int)
    _P  = expit(rbm.beta*(np.matmul(_H, np.transpose(_W)) + _b))
    _V = (Vb_U <= _P).astype(int)
    assert(0 == np.linalg.norm(V - _V))
    
#
# Test decreasing step size
#
def test_tc10():
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
    # Now train the model - run 2 of 10 iterations
    # in one epoch
    #
    rbm.train(V, iterations=2, epochs=5, step = 0.1)
    #
    # Check global_step
    #
    global_step_v =  [v for v in tf.global_variables() if v.name == "globalStep:0"][0]
    global_step = rbm.session.run(global_step_v)
    #
    # Get step size currently used
    #
    #print([t.name for op in tf.get_default_graph().get_operations() for t in op.values()])
    step = run_operation(rbm, "delta/step")
    #
    # The call to run has set this to the new value corresponding to global_step
    #
    _step = 0.05 * (1.0 - (1.0*global_step)/(1.0*5*2))
    assert(np.abs(step - _step) < epsilon)


#
# Test the CPU based version of the PCD algorithm
#
def test_tc11():
    #
    # Fix a random seed to produce a deterministic result
    #
    np.random.seed(25)
    #
    # Create machine
    #
    rbm = RBM.PCD.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # Now train the model - run 2 of 10 iterations
    # in one epoch
    #
    rbm.train(V, iterations=2, epochs=5, step = 0.1)
    #
    # Expected results - captured from a reference run
    #
    _W = [[-0.04370055, -0.03594863],[ 0.04134263, 0.04370914],  [-0.05932903, -0.05216718]]
    _b = [[-0.09 , 0.1 , -0.1 ]]
    _c = [[ -2.20123998e-05 , -3.23122517e-04]]
    error = np.linalg.norm(rbm.W - _W)
    assert(error < epsilon)
    error = np.linalg.norm(rbm.b - _b)
    assert(error < epsilon)
    error = np.linalg.norm(rbm.c - _c)
    assert(error < epsilon)
    
    
    

#
# Test the CPU based version of the CD-1 algorithm
#
def test_tc12():
    #
    # Fix a random seed to produce a deterministic result
    #
    np.random.seed(25)
    #
    # Create machine
    #
    rbm = RBM.CD.CDRBM(visible=3, hidden=2, beta = 1.0)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # Now train the model - run 2 of 10 iterations
    # in one epoch
    #
    rbm.train(V, iterations=2, epochs=5, step = 0.1)
    #
    # Expected results - captured from a reference run
    #
    _W = [[ 0.00228273 , 0.0102689 ],  [-0.00815664, -0.00585623],  [-0.05911978, -0.05201988]]
    _b =[[ 0.0,   0.,  -0.1]]
    _c = [[  2.39211229e-04,   5.55807324e-05]]
    error = np.linalg.norm(rbm.W - _W)
    assert(error < epsilon)
    error = np.linalg.norm(rbm.b - _b)
    assert(error < epsilon)
    error = np.linalg.norm(rbm.c - _c)
    assert(error < epsilon)
    
#
# Test one entire training step with 32 floating point precision
#
def test_tc13():
    #
    # Create machine and build model
    #
    weight_decay = 0.001
    batch_size = 4
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=4, precision = 32)
    rbm.prepare_training(batch_size=4, weight_decay = weight_decay, total_steps = 10, initial_step_size = 0.05)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(4,3))
    V[0,1] = 1
    V[1,2] = 1
    V[2,0] = 1
    V[3,1] = 1
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
    db_op = tf.get_default_graph().get_operation_by_name("delta/db")
    dc_op = tf.get_default_graph().get_operation_by_name("delta/dc")
    dW_op = tf.get_default_graph().get_operation_by_name("delta/dW")
    _, ndw, H_U, Nb_U, neg, pos, db, dc, dW = rbm.session.run([rbm.tf.trainingStep,
                                            rbm.tf.norm_dW,
                                            H_U_op.outputs[0], 
                                            Nb_U_op.outputs[0],
                                            rbm.tf.neg, rbm.tf.pos,
                                            db_op.outputs[0], 
                                            dc_op.outputs[0], 
                                            dW_op.outputs[0]],
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
    _db = 0.05 *rbm.beta*np.sum(V - N, 0) / float(batch_size)
    error = np.linalg.norm(_db - db)
    assert(error < 0.01)
    _E = expit(rbm.beta*(np.matmul(V, rbm.W) + rbm.c))
    _dc = 0.05*rbm.beta*np.sum(_E - _Eb, 0) / float(batch_size)
    error = np.linalg.norm(_dc - dc)
    assert(error < 0.001)
    #
    # Check weight updates
    #
    _dW = 0.05*(rbm.beta*(pos -neg)  - weight_decay*rbm.W) / float(batch_size)
    error = np.linalg.norm(_dW - dW)
    assert(error < 0.01)
    #
    # Finally make sure that the parameters have actually been updated
    #
    _W = rbm.W + _dW
    _b = rbm.b + _db
    _c = rbm.c + _dc
    W, b, c = rbm.session.run([rbm.tf.W, rbm.tf.b, rbm.tf.c])
    assert(np.linalg.norm(W - _W) < 0.008)
    
#
# Test one sampling step in detail, using precision 32 bit
#
def test_tc14():
    #
    # Create machine
    #
    rbm = RBM.PCDTF.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1, precision = 32)
    rbm.prepare_training(batch_size = 2, weight_decay = 0.001, total_steps = 10, initial_step_size = 0.05)
    # Prepare input data
    #
    V = np.zeros(shape=(2,3))
    V[0,1] = 1
    V[1,0] = 1
    #
    # Now train the model
    #
    for _ in range(10):
        rbm.train(V, iterations=1, epochs=10)
    #
    # Get the parameters
    #
    rbm.postTraining()
    params = rbm.getParameters()
    _W = params['W']
    _b = params['b']
    _c = params['c']
    initial = np.zeros(shape=(2,3))
    initial[0,1] = 1
    initial[1,2] = 1
    #
    # Build sampling model
    #
    tf.reset_default_graph()
    rbm.build_sampling_model(sample_size=2)
    #
    # Prepare session
    #
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    rbm.tf.V.load(initial, session)
    rbm.tf.W.load(_W, session)
    rbm.tf.b.load(_b, session)
    rbm.tf.c.load(_c, session)
    #
    # Get references to the two random variables used for sampling
    #
    Vb_U_op = tf.get_default_graph().get_operation_by_name("sampling/Vb_U")
    H_U_op = tf.get_default_graph().get_operation_by_name("sampling/H_U")
    _, Vb_U, H_U = session.run([rbm.tf.runGibbsStep, 
                                Vb_U_op.outputs[0], 
                                H_U_op.outputs[0]])
    #
    # and get result of sampling
    #
    V = session.run(rbm.tf.V)
    #
    # Now sample one step manually and compare
    #
    _E = expit(rbm.beta*(np.matmul(initial, _W) + _c))
    _H = (H_U <= _E).astype(int)
    _P  = expit(rbm.beta*(np.matmul(_H, np.transpose(_W)) + _b))
    _V = (Vb_U <= _P).astype(int)
    assert(0 == np.linalg.norm(V - _V))
    
#
# Test the CPU based version of the CD-1 algorithm,
# using 32 bit precision
#
def test_tc15():
    #
    # Fix a random seed to produce a deterministic result
    #
    np.random.seed(25)
    #
    # Create machine
    #
    rbm = RBM.CD.CDRBM(visible=3, hidden=2, beta = 1.0, precision=32)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # Now train the model - run 2 of 10 iterations
    # in one epoch
    #
    rbm.train(V, iterations=2, epochs=5, step = 0.1)
    #
    # Expected results - captured from a reference run
    #
    _W = [[ 0.00228273 , 0.0102689 ],  [-0.00815664, -0.00585623],  [-0.05911978, -0.05201988]]
    _b =[[ 0.0,   0.,  -0.1]]
    _c = [[  2.39211229e-04,   5.55807324e-05]]
    error = np.linalg.norm(rbm.W - _W)
    assert(error < epsilon)
    error = np.linalg.norm(rbm.b - _b)
    assert(error < epsilon)
    error = np.linalg.norm(rbm.c - _c)
    assert(error < epsilon)
    
#
# Test the CPU based version of the PCD algorithm
# precision = 32
#
def test_tc16():
    #
    # Fix a random seed to produce a deterministic result
    #
    np.random.seed(25)
    #
    # Create machine
    #
    rbm = RBM.PCD.PCDRBM(visible=3, hidden=2, beta = 1.0, particles=1, precision =32)
    #
    # Prepare input data
    #
    V = np.zeros(shape=(1,3))
    V[0,1] = 1
    #
    # Now train the model - run 2 of 10 iterations
    # in one epoch
    #
    rbm.train(V, iterations=2, epochs=5, step = 0.1)
    #
    # Expected results - captured from a reference run
    #
    _W = [[-0.04370055, -0.03594863],[ 0.04134263, 0.04370914],  [-0.05932903, -0.05216718]]
    _b = [[-0.09 , 0.1 , -0.1 ]]
    _c = [[ -2.20123998e-05 , -3.23122517e-04]]
    error = np.linalg.norm(rbm.W - _W)
    assert(error < epsilon)
    error = np.linalg.norm(rbm.b - _b)
    assert(error < epsilon)
    error = np.linalg.norm(rbm.c - _c)
    assert(error < epsilon)

