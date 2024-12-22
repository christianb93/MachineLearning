# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#
# Entropy of a composite system 
# consisting of two ideal gases with particle
# numbers N1 and N2 
# U is the total energy, X is the energy of the first subsystem
# C = cR is the common prefactor
#
def S(U, X, C = 1.0, N1 = 1.0, N2 = 1.0):
    return C*N2 * np.ma.log((U-X)*np.float_power(X, N1/N2)).filled(np.nan)

#
# Get equilibrium state for a given 
# value of U. Return X value
# and entropy
#
def get_XS(U, C = 1.0, N1 = 1.0, N2 = 1.0):
    _X = N1 / (N1 + N2) * U
    return _X, S(U,_X, N1 = N1, N2 = N2, C = C)


#
# Solve for the total energy given S and X
#
def get_U(S, X, C = 1.0, N1 = 1.0, N2 = 1.0):
    return X + np.exp(S/(N2*C)) * np.power(X, - N1/N2)

#######################################
# Main
#######################################
 
C = 1.2
N1 = 5.5
N2 = 3.0

U = np.arange(5, 12, 0.1)
X = np.arange(1, 12, 0.1)
A, B =  np.meshgrid(U,X)
Z = S(A,B, N1 = N1, N2 = N2, C = C)

#
# Plot values of the entropy
#
fig = plt.figure(figsize=(10,8))
ax = fig.gca(projection='3d')
ax.set_xlabel("U")
ax.set_ylabel("U1")
ax.set_zlabel("S")
#ax.set_xticks([])                               
#ax.set_yticks([])                               
#ax.set_zticks([])
ax.set_xlim3d(5, 13)
ax.set_ylim3d(0, 13)
ax.set_zlim3d(0, 20)
ax.plot_surface(A,B,Z, color="0.98", rstride=1, cstride=1, alpha = 0.5)

#
# Now add some 2-dimensional plots for fixed
# values of U 
#
for U0 in np.arange(5.0,9.0, 1.0):
    X = np.arange(1.0, 11, 0.1)
    U = np.full((len(X)), U0)
    ax.plot(U, X, S(U, X, N1 = N1, N2 = N2, C = C), 'b')
    
#
# and now add some equilibrium points
U = np.arange(5.0, 8.1, 0.1)
X, S = get_XS(U, N1 = N1, N2 = N2, C = C)
ax.plot(U, X, S, 'k--')

#
# Now add a curve for fixed S
# Use the highest value for S
# obtained in the last exercise
#
S0 = S[len(S)-1]
X = np.arange(3.0, 10, 0.2)
S = np.full((len(X)), S0)
U = get_U(S,X, C=C, N1=N1, N2=N2)
ax.plot(U, X, S, 'b--')

#
# Show everything
#
ax.view_init(elev=45, azim=150)
plt.savefig("EntropyPlot.png")
plt.show()
