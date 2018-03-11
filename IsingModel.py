#################################################################
# 
# Ising model simulation
#
# If you run this without an X server, set MPLBACKEND to AGG:
# export MPLBACKEND="AGG"
# MIT license
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
#
#################################################################

from __future__ import print_function  
import argparse
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tempfile

class IsingModel:
    
    #
    # Restart but do not delete state
    # 
    def restart(self, T = 1):
        # States visited in total
        self.visited_states = 1
        # States visited so far after burn_in
        self.recorded_states = 0
        # Inverse temperature
        self.beta = 1 / T
        # and means
        self.meanE = 0
        self.meanEsquare = 0
        self.meanM = 0

    
    #
    # Initialize the model
    #
    def __init__(self, N = 10, d = 1, T = 1, rows = 10, cols = 10):
        # Dimension
        self.d = d
        # Number of spins and geometry
        self.N = N
        self.rows = rows
        self.cols = cols        
        if self.d == 2:
            if (rows*cols != N):
                raise ValueError("Geometry (rows = ", rows," cols = ",cols,") not compatible with N=", N)
        # Set the state to some random value
        self.s =2*np.random.randint(0,2,size=N) - 1
        self.E = self.energy()
        self.M = np.average(self.s)
        # and restart counter
        self.restart(T=T)
        
        
    #
    # Given a location i, determine all nearest
    # neighbors. We use free boundary conditions
    #
    def get_nearest_neighbours(self,i):
        if ((i >= self.N) or (i < 0)):
            raise ValueError("Index ",i,"out of range")
        NN = []
        if self.d == 1:
            if (i == 0):
                NN.append(1)
            elif (i == N-1):
                NN.append(N-2)
            else:
                NN.append(i-1)
                NN.append(i+1)
        elif self.d == 2:
            r = int(i / self.cols)
            c = int(i % self.cols)
            #
            # Go through all four neighbors and check whether they are on 
            # the grid
            #
            iu = (r-1)*self.cols + c
            if r > 0:
                NN.append(int(iu))
            id = (r+1)*self.cols + c
            if r < self.rows - 1:
                NN.append(int(id))
            il = i - 1
            if (c > 0):
                NN.append(int(il))
            ir = i + 1
            if (c < self.cols - 1):
                NN.append(int(ir))
        else:
            raise ValueError("Value",self.d,"for dimension not supported")
        return NN
    
    #
    # Calculate energy of current state
    #
    def energy(self):
        E = 0
        for i in range(self.N):
            NN = self.get_nearest_neighbours(i)
            for j in NN:
                E -= (1/2) * self.s[i] * self.s[j]
        return E
        
    #
    # Calculate the local field at position i
    #
    def local_field(self, i):
        B = 0
        NN = self.get_nearest_neighbours(i)
        for j in NN:
            B += self.s[j]
        return B
        
    #
    # Run a given number of Gibbs sampling steps
    #
    def sample(self, steps, record = 1):
        for _ in range(steps):
            #
            # Choose a coordinate at random
            #
            i = np.random.randint(0,self.N)
            #
            # Calculate local field and change of energy
            #
            DE = -2*self.local_field(i)
            #
            # and conditional probability
            # 
            P = 1 / (1 + np.exp(self.beta*DE))
            #
            # Save old state and update
            #
            old = self.s[i]
            U = np.random.uniform()
            if (U <= P):
                self.s[i] = 1
            else:
                self.s[i] = -1
            #
            # Update statistics
            #
            if (old != self.s[i]):
                self.E -= old*DE
            self.visited_states += 1
            if (record == 1):
                self.recorded_states += 1
                self.meanE += self.E
                self.meanEsquare += self.E**2
                self.M += self.s[i] - old
                self.meanM += self.M
            
    #
    # Plot current state into an axis object
    #
    def plot(self,ax):
        if (self.d == 1):
            ax.imshow(self.s.reshape(1,self.N), "binary")
        elif (self.d == 2):
            ax.imshow(self.s.reshape(self.rows, self.cols), "binary")
        else:
            raise ValueError("Value",self.d,"for dimension not supported")

    
    #
    # Return statistics:
    # mean value of energy
    # mean square value of energy
    #
    def statistics(self):
        return (self.meanE / self.recorded_states, 
                self.meanEsquare  / self.recorded_states,
                self.meanM / self.recorded_states)
        
        
###############################################################
# Main
###############################################################
        
        
#
# Parse arguments. The defaults are small on purpose
# 
parser = argparse.ArgumentParser()
parser.add_argument("--d", 
                    type=int,
                    default=2,
                    help="Dimension")
parser.add_argument("--steps", 
                    type=int,
                    default=100,
                    help="Number of steps per epoch")
parser.add_argument("--epochs", 
                    type=int,
                    default=100,
                    help="Number of epochs")
parser.add_argument("--burn_in", 
                    type=int,
                    default=30,
                    help="Number of burn-in epochs")
parser.add_argument("--dT",
                    type=float,
                    default=0.2,
                    help="Temperature interval")
parser.add_argument("--Tmax",
                    type=float,
                    default=5.0,
                    help="Maximal temperature")
parser.add_argument("--Tmin",
                    type=float,
                    default=0.1,
                    help="Minimum temperature")
parser.add_argument("--N", 
                    type=int,
                    default=400,
                    help="Number of particles")
parser.add_argument("--rows", 
                    type=int,
                    default=20,
                    help="Rows of lattice for d = 2")
parser.add_argument("--cols", 
                    type=int,
                    default=20,
                    help="Columns of lattice for d = 2")
parser.add_argument("--show",
                    type=int,
                    default=0,
                    help="Display results")
parser.add_argument("--plotM",
                    type=int,
                    default=0,
                    help="Plot magnetization")
args = parser.parse_args()
print(args)
start_time = time.time()
print("Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

epochs = args.epochs
burn_in = args.burn_in
steps = args.steps
N = args.N
rows = args.rows
cols = args.cols
d = args.d
dT = args.dT
Tmax = args.Tmax
Tmin = args.Tmin
plotM = args.plotM
T = np.arange(Tmax,Tmin, -dT)

#
# Geometry of plots for the model itself. For d=1, we plot one
# bar per temperature. For d = 2, we plot one square per temperature
# and organize the plots into a matrix
#
fig = plt.figure(figsize=(15,15))
plot = 1
oned_plots = len(T)
twod_plots = len(T)
twod_cols = int(1 / dT)
twod_rows = int(twod_plots / twod_cols)
if (twod_rows * twod_cols < twod_plots):
    twod_rows += 1

mE = []  
vE = []
mM = []
IM = IsingModel(N=N, d=d, T = 1, cols = cols, rows = rows)
for _T in T:
    #
    # Reset parameters, but not state
    #
    IM.restart(T = _T)
    _steps = steps
    print("Running simulation for d = ", d, "N=", N," with ",epochs*_steps,"iterations for temperature T=",_T)
    for _ in range(epochs):
        if (_ < burn_in):
            record = 0
        else:
            record = 1
        IM.sample(_steps, record)

    _mE, _mEsquare, _mM = IM.statistics()    
    _vE = _mEsquare - _mE**2
    mE.append(_mE)
    vE.append(_vE)
    mM.append(_mM)
    if d == 1:
        ax = fig.add_subplot(oned_plots,1,plot)
    if d == 2:
        ax = fig.add_subplot(twod_rows, twod_cols,plot)
    plot += 1
    ax.set_yticks([],[])
    ax.set_xticks([],[])
    IM.plot(ax)    
    
# fig.tight_layout()
tmp = tempfile.mktemp()
outfile = tmp + "_IsingPartI.png"
print("Saving simulation results part I to ",outfile)
fig.savefig(outfile)
if args.show == 1:
    plt.show()

if plotM == 1:
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(2,2,1)
else:
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,2,1)
ax.set_xlabel("T")
ax.set_ylabel("E")
ax.plot(T,mE, "b")

#
# Plot theoretical prediction in case d == 1
#
if d == 1:
    tE = - np.tanh(1/T)*N
    ax.plot(T, tE, "y" )

# 
# Now plot variance
# 
if plotM == 1:
    ax = fig.add_subplot(2,2,2)
else:
    ax = fig.add_subplot(1,2,2)
ax.set_xlabel("T")
ax.set_ylabel("var(E)")
ax.plot(T,vE, "b")

if d == 1:
    tV = N*(1-np.tanh(1/T))
    ax.plot(T,tV, "y")
    
#
# And finally magnetization
#    
if plotM == 1:
    ax = fig.add_subplot(2,1,2)
    ax.set_xlabel("T")
    ax.set_ylabel("M")
    ax.plot(T,mM, "b")


fig.tight_layout()

outfile = tmp + "_IsingPartII.png"
print("Saving simulation results part I to ",outfile)
fig.savefig(outfile)
if args.show == 1:
    plt.show()

#
# Save description and data
# 
outfile = tmp + "_Ising.txt"
f= open(outfile, "w")
print("N = ", N, "d=", d, "steps = ", steps, "dT = ", dT, "burn_in = ", burn_in, "epochs = ", epochs, file=f)
print("Rows = ", rows, "Cols = ", cols, file=f)
print("Full arguments: ", args, file=f)
print("Energy means:", file=f)
print(mE, file=f)
print("Magnetization:", file=f)
print(mM, file=f)
print("Variance:", file=f)
print(vE, file=f)
print("Temperatures:", file=f)
print(T, file=f)
f.close()
print("Saved simulation description and results in ",outfile)
    
end_time = time.time()
run_time = end_time - start_time
print("End time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("Run time: ", str(datetime.timedelta(seconds=int(run_time))))
