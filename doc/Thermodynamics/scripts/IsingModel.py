#########################################
# Draw a few diagrams for an Ising model
#########################################

import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy
import scipy.optimize


#
# Evaluate the consistency condition, i.e. 
# cc(Tb, M, B) = M - tanh(1/Tb * (M + B / Jq))
#
# Here Tb is the reduced temperature, q is the
# number of nearest neighbors and J is the interaction strength
#
def cc(M, Tb, B, q = 4, J = 1.0):
    return M - np.tanh(1.0 / Tb * (M - B / (1.0 * J * q)))


#
# Search zeros in an array, i.e. look for indices where the sign
# switches, and return them as an arry
#  
def find_roots(y):
    old_sign = np.sign(y[0])
    roots = []
    for i, _y in enumerate(y):
        if np.sign(_y) != old_sign:
            old_sign = np.sign(_y)
            roots.append(i)
    return roots

#
# Create a plot for the values of the consistency condition and
# mark the zeros
#
def do_plot(Tb, B, ax):
    ax.set_xlabel("Magnetization M (B =" + "{:.1f}".format(B) + ")")
    M = np.arange(-1, 1, 0.01)
    values = cc(M, Tb, B = B)
    ax.plot(M, values)
    ax.plot(M, 0.0 * values, "k--")
    #
    # Determine the zeros
    #
    roots = find_roots(values)
    for r in roots:
        ax.plot(M[r], values[r], "rD")


#
# Calculate the variational Helmholtz potential as a function of 
# m for B = 0
#
def Fv(Tb, M, J = 1.0, q = 4, N = 1):
    return - 1.0 * J * q  * Tb * N * np.log(2 * np.cosh(M / Tb)) + 0.5 * J * q * M**2 * N

#
# Solve the consistency condition for a given temperature
# and B = 0
# 
def solve_cc(Tb):
    #
    # As tanh is in [-1, 1], we only need to search that range
    # and we restrict our search to the positive range
    # 
    M = np.arange(.01, 1.1, 0.001)
    values = cc(M, Tb, B = 0.0)
    roots = find_roots(values)
    if len(roots) == 0:
        print("Did not find a root")
        exit(1)
    root = M[roots[0]]
    return root

#
# First plot the consistency condition as a function of M
# for fixed B.
#
fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(3, 1, 1)
ax1 = fig.add_subplot(3, 1, 2)
ax2 = fig.add_subplot(3, 1, 3)
Tb = 0.8
do_plot(Tb = Tb, B = 0.0, ax = ax0)
do_plot(Tb = Tb, B = 0.2, ax = ax1)
do_plot(Tb = Tb, B = 0.6, ax = ax2)
plt.savefig("IsingModelMagnetization.png")

#
# Now plot the variational Helmholtz potential for B = 0
#
fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(1, 1, 1)
M = np.arange(-1.5, 1.5, 0.05)
ax0.plot(M, Fv(0.6, M))
ax0.plot(M, Fv(0.8, M))
ax0.plot(M, Fv(1.0, M))
ax0.plot(M, Fv(1.2, M))
ax0.set_xlabel("Magnetization")
ax0.set_ylabel("Variational free energy")
plt.savefig("IsingModelHelmholtzEnergy.png")

#
# Now plot the magnetizations close to Tc
#
fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(1, 1, 1)
Tb = np.arange(0.8, 0.9995, 0.0005)
roots = [solve_cc(_Tb) for _Tb in Tb]
Tb = np.append(Tb, 1.0)
roots = np.append(roots, 0.0)
ax0.plot(Tb, roots, "b")
roots = [- 1.0 * _r for _r in roots]
ax0.plot(Tb, roots, "b")
ax0.set_xlabel("Temperature")
ax0.set_ylabel("Net magnetization")
plt.savefig("IsingModelCriticalTemperature.png")

plt.show()
