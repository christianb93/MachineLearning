###########################################
# Hopfield network
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
###########################################

import numpy as np
import matplotlib.pyplot as plt 
import tempfile
import argparse


############################################
#
# Some patterns that we use for testing
#
############################################

strings = []

strings.append("""
..X..
.X.X.
X...X
.X.X.
..X..""")

strings.append("""
..X..
..X..
..X..
..X..
..X..""")

strings.append("""
.....
.....
XXXXX
.....
.....""")

strings.append("""
X....
.X...
..X..
...X.
....X""")

strings.append("""
....X
...X.
..X..
.X...
X....""")

############################################
#
# Some utility functions
#
############################################


#
# Convert a string as above into a 
# 5 x 5 matrix
#
def string_to_matrix(s):
    x = np.zeros(shape=(5,5), dtype=float)
    for i in range(len(s)):
        row, col = i // 5, i % 5
        x[row][col] = -1 if s[i] == 'X' else 1
    return x

#
# and back
#
def matrix_to_string(m):
    s = ""
    for i in range(5):
        for j in range(5):
            s = s + ('X' if m[i][j] < 0 else '')
        s = s + chr(10)
    return s
    

class HopfieldNetwork:
    
    #
    # Initialize a Hopfield network with N 
    # neurons
    #
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N,N))
        self.s = np.zeros((N,1))
    
    #
    # Apply the Hebbian learning rule. The argument is a matrix S
    # which contains one sample state per row
    #
    def train(self, S):
        self.W = np.matmul(S.transpose(), S)
    
    #
    # Run one simulation step
    #
    def runStep(self):
        i = np.random.randint(0,self.N)
        a = np.matmul(self.W[i,:], self.s)
        if a < 0:
            self.s[i] = -1
        else:
            self.s[i] = 1
    
    #
    # Starting with a given state, execute the update rule
    # N times and return the resulting state
    #
    def run(self, state, steps):
        self.s = state
        for i in range(steps):
            self.runStep()
        return self.s
        
        
############################################
#
# Parse arguments
#
#############################################        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories", 
                    type=int,
                    default=3,
                    help="Number of patterns to learn")
    parser.add_argument("--epochs", 
                    type=int,
                    default=6,
                    help="Number of epochs")
    parser.add_argument("--iterations", 
                    type=int,
                    default=20,
                    help="Number of iterations per epoch")                    
    parser.add_argument("--errors",
                    type=int,
                    default=5,
                    help="Number of error that we add to each sample")
    parser.add_argument("--save",
                    type=int,
                    default=0,
                    help="Save output")
    return parser.parse_args()
    

############################################
#
# Main
#
#############################################


#
# Read parameters
#
args = get_args()

#
# Number of epochs. After each
# epoch, we capture one image
#
epochs = args.epochs

#
# Number of iterations
# per epoch
#
iterations = args.iterations

#
# Number of bits that we flip in each sample
#
errors = args.errors

#
# Number of patterns that we try to memorize
#
memories = args.memories

# 
# Init network
#
HN = HopfieldNetwork(5*5)

#
# Prepare sample data and train network
#

M = []
for _ in range(memories):
    M.append(string_to_matrix(strings[_].replace(chr(10), '')).reshape(1,5*5))
S = np.concatenate(M)
HN.train(S)

#
# Run the network and display results
#

fig = plt.figure()
    
for pic in range(memories):

    state = (S[pic,:].reshape(25,1)).copy()
    #
    # Display original pattern
    #
    ax = fig.add_subplot(memories,epochs + 1, 1+pic*(epochs+1))
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.imshow(state.reshape(5,5), "binary_r")
    #
    # Flip a few bits
    #
    state = state.copy()
    for i in range(errors):
        index = np.random.randint(0,25)
        state[index][0] = state[index][0]*(-1)
    #
    # Run network and display the current state
    # at the beginning of each epoch
    #
    for i in range(epochs):
        ax = fig.add_subplot(memories,epochs + 1, i+2+pic*(epochs+1))
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.imshow(state.reshape(5,5), "binary_r")
        state = HN.run(state, iterations)

if 1 == args.save:        
    outfile = tempfile.mktemp() + "_Hopfield.png"
    print("Using outfile ", outfile)
    plt.savefig(outfile)
plt.show()
