###############################################
# 
# Examples for the Metropolis-Hastings         
# algorithm                                   
# 
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
###############################################

from __future__ import print_function  

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats



###############################################
# The probability density from which we want  
# to sample                                   
###############################################

def p(x, target="cauchy"):
    if target == "cauchy":
        #
        # Cauchy with mode 0 and scale 1
        #
        gamma = 1
        return 1/(np.pi *gamma) * (gamma*gamma / (x*x + gamma*gamma))
    else:
        if (x <= 0):
            return 0
        return np.exp(-x)
    

###############################################
# The proposal                         
###############################################
    
def propose(x, std_dev=1.0, mode='symmetric'):
    if mode == 'symmetric':
        #
        # normal distribution with mean x 
        #
        return np.random.normal(loc=x, scale=std_dev)
    elif mode == 'independent':
        return np.random.normal(loc=1, scale=std_dev)


###############################################
# The proposal density
###############################################
    
def q(y, std_dev=1.0):
    #
    # Assume mode = independent
    #
    return scipy.stats.norm.pdf((y -1) / std_dev) / std_dev

###############################################
# Get parameters
###############################################
parser = argparse.ArgumentParser()
parser.add_argument("--mode",
                   default="symmetric",
                   choices=["symmetric", "independent"])
parser.add_argument("--target",
                    default="cauchy",
                    choices=["cauchy","exp"])
parser.add_argument("--steps", 
                    type=int,
                    default=10000,
                    help="Number of steps")
parser.add_argument("--outfile", 
                    default=None,
                    help="Save visualization as PNG")
parser.add_argument("--gap",
                    type=int,
                    default=10,
                    help="Gap used for subsampling")
parser.add_argument("--std_dev",
                    type=float,
                    default=1.0,
                    help="Standard deviation of proposal distribution")
parser.add_argument("--show",
                    type=int,
                    default=1,
                    help="Show output")
parser.add_argument("--save_mem",
                    type=int,
                    default=0,
                    help="Skip some functions to save memory")
args = parser.parse_args()



###############################################
# The actual algorithm
###############################################
    
integral_sin = 0
integral_cos = 0
integral_sin_values = []
integral_cos_values = []


print("Running chain with", args.steps, "steps")
print("Std_dev = ", args.std_dev)
print("Gap = ", args.gap)
chain = []
#
# Choose a starting point
#
X = 1
reject = 0.0
for n in range(args.steps):
    Y = propose(X, std_dev = args.std_dev)
    U = np.random.uniform()
    if args.mode == 'symmetric':
        alpha = p(Y, target=args.target) / p(X, target=args.target)
    elif args.mode == 'independent':
        qy = q(Y, std_dev = args.std_dev)
        qx = q(X, std_dev = args.std_dev) 
        if qy == 0:
            qy = 0.0000000001
        alpha = p(Y, target=args.target) / p(X, target=args.target) * qx / qy
    if (U <= alpha):
        X = Y
    else:
        reject += 1
    if (n > 0) and (0 == n % 100000):
        print("Step ",n)
    # 
    # Calculate some test integrals
    #
    integral_sin += np.sin(X)
    integral_cos += np.cos(X)
    if args.save_mem==0:
        integral_sin_values.append(integral_sin/(n+1))
        integral_cos_values.append(integral_cos/(n+1))
        chain.append(X)


    
##############################################
# Display results
##############################################

#
# Acceptance rate
# 
print("Rejects: ", reject)
print("Acceptance rate:", 1.0 - (reject /(args.steps*1.0)))


#
# Integral of sin(x) and cos(x)
#
integral_sin = integral_sin / args.steps
integral_cos = integral_cos / args.steps

#
# Get true values
#
if args.target == "cauchy":
    integral_sin_target = 0
    integral_cos_target = 0.3678
elif args.target == "exp":
    integral_sin_target = 0.5
    integral_cos_target = 0.5

print("Integral of sin (true value is",integral_sin_target,"): ", integral_sin)
print("Integral of cos (true value ", integral_cos_target,"):", integral_cos)


#
# If there is nothing to plot we are done
#
if args.save_mem==1:
    exit(0)

#
# First we plot the full chain
#
print("Plotting trace")
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(3,1,1)
ax.plot(chain)

#
# Next we create a real sample and plot this
#
sample = []
burn_in = args.steps *0.2
print("Using burn in of", burn_in)
length = 0
for n in range(args.steps):
    if (n > burn_in) and (n % args.gap == 0):
        sample.append(chain[n])
        length += 1
       
print("Created sample of length",length)


#
# Fix some parameters used for plotting
# 
bins = 20
rwidth = 0.9
hist_range = (-5,5)

#
# Get values of pdf of target distribution
#
pdf = []
for x in np.arange(-5, 5, 0.1):
    pdf.append(p(x, target=args.target))

print("Creating histogramm")
ax = fig.add_subplot(3,2,3)
ax.set_xlim(left=-5, right=5)
ax.set_ylim(bottom=0, top=1.0)
ax.hist(sample,bins=bins, normed=1, range=hist_range, rwidth=rwidth)
ax.plot(np.arange(-5, 5, 0.1), pdf)

print("Calculating autocorrelations")
ax = fig.add_subplot(3,2,4)
#
# Calculate the autocorrelation with lags up to maxlag
#
if args.steps < 5000:
    maxlag = args.steps
else:
    maxlag = 5000
corr = []
lags = []
for lag in np.arange(0,maxlag, 50):
    lags.append(lag)
    corr.append(np.corrcoef(chain[lag:], chain[:args.steps - lag])[0,1])
ax.plot(lags, corr, "r+")

#
# Show how integral converges
# 
ax = fig.add_subplot(3,1,3)
ax.plot(integral_sin_values)
ax.plot(integral_cos_values)



#
# Do Kolmogorov-Smirnoff test against target distribution
#
if args.target == "cauchy":
  comp_sample = np.random.standard_cauchy(length)
else:
  comp_sample = np.random.exponential(size=length)
D,p_value = scipy.stats.ks_2samp(sample, comp_sample)
print("P-value:", p_value)
print("KS statistics:", D)



if args.outfile:
    plt.savefig(args.outfile)
if args.show==1:
    plt.show()


