#############################################
#
# Random walk on a circle                   
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
#############################################

import numpy as np
import matplotlib.pyplot as plt
#import tabulate


##############################################
# Draw from a finite distribution defined
# by a vector p with elements adding up
# to one. We return a number between
# 1 and the number of elements of p
# and i is returned with probability p[i-1]
##############################################
def draw(p):
    u = np.random.uniform()
    x = 0
    n = 1
    for i in p:
        x = x + i
        if x >= u:
            return n
        n += 1
    return i



##############################################
# Build the transition matrix
# We stay at position i with probability
# 1 - p and move to i-1 with probability
# p/2 and to i+2 with probability p/2
##############################################
def build_kernel(N=3, p = 1.0):
    K = np.zeros((N,N), dtype=float)
    for i in range(N):
        left = (i - 1) % N
        right = (i + 1) % N
        K[i,left] = p/ 2.0
        K[i,right] = p / 2.0
        K[i,i] = 1.0 - p
    return K


##############################################
# Simulate a random walk on the circle with
# N points and probability to move p
# Return the simulated chain
##############################################
def simulate_chain(N, p, steps=100, start=5):
    chain = []
    x = start % N
    chain.append(x)
    for i in range(steps):
        x = (x  + draw([p/2.0, 1.0 - p, p / 2.0]) - 2) % N
        chain.append(x)
    return chain

##############################################
# Main
##############################################
    

#
# Build kernel
#
N = 10
p = 0.8
K = build_kernel(N=N, p=p)

# print("Transition matrix: ", K)
# print(tabulate.tabulate(K, tablefmt="latex", floatfmt=".2f"))


#
# Run a sample path and plot the resulting 
# histograms
#
fig = plt.figure(figsize=(10,8))
batch_size = 100
grid_x = 4
grid_y = 4
chain = simulate_chain(N=N, steps=batch_size*grid_x*grid_y, start=5, p=p)
for i in range(grid_x*grid_y):
    ax = fig.add_subplot(grid_x, grid_y,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.hist(chain[0:(i+1)*batch_size], rwidth=0.8)


#plt.savefig("/tmp/CircleRandomWalk.png")
plt.show()

fig = plt.figure(figsize=(10,8))
grid_x = 4
grid_y = 4
P = np.identity(N)
for i in range(grid_x*grid_y):
    ax = fig.add_subplot(grid_x, grid_y,i+1)
    ax.set_xlim(0,N-1)
    ax.set_xticks([])
    ax.set_ylim(0,1)
    ax.set_yticks([])
    # Plot current distribution for some starting points
    ax.plot(P[4,:],"g--")
    ax.plot(P[7,:],"b--")
    ax.plot(P[1,:],"r--")
    P = np.matmul(K,P)


#plt.savefig("/tmp/CircleRandomWalkMatrixPowers.png")
plt.show()
