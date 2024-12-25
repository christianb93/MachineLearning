#
# Plot the energy levels of a two-state system for 
# various values of B
#

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

def U(B, beta, N = 50):
    return - N * B * np.tanh( beta * B)


def F(B, beta, N = 50):
    return - N * (1.0 / beta) * np.log(2.0 * np.cosh(beta * B))

#
# Calculate the multiplicity of an energy level
# Nplus = spins pointing upwards, so that E = - B NPlus + B Nminus 
#
def omega(Nplus, B = 1.0, N = 50):
    Nminus = N - Nplus
    if Nminus == 0:
        return 1
    if Nplus == 0:
        return 1
    return special.factorial(N) / (special.factorial(Nplus) * special.factorial(Nminus))

#
# Calculate the (non-normalized) probability of an energy level
# E = - 2 B Nplus - E0 
# where E0 = - BN is the lowest energy level 
#
def pE(Nplus, beta, B = 1.0, N = 50):
    E = - 2 * B * Nplus + B * N
    return omega(Nplus, B, N) * np.exp(- beta * E)

#
# Calculate the partition function
#
def Z(beta = 1.0, B = 1.0, N = 50):
    return 2**N * np.cosh(beta * B)**N


#
# Determine various characteristic energies and return the results
#
def calc_energies(N = 50, beta = 1.0, B = 1.0):
    Nplus = range(0, N)
    E0 = - B * N
    _E = [- 2.0 * B * _Nplus  - E0 for _Nplus in Nplus]
    _P = [pE(_Nplus, beta, B = B, N = N)  for _Nplus in Nplus ]
    #
    # Determine the most likely energy
    #
    pMax = 0.0
    mlE = 0.0
    for i, p in enumerate(_P):
        if p > pMax:
            mlE = _E[i] 
            pMax = p
    #
    # Determine U and F
    #
    _U = U(B = B, beta = beta, N = N)
    _F = F(B = B, beta = beta, N = N)
    return [mlE, _U, E0, _F]

fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(2, 1, 1)
ax1 = fig.add_subplot(2, 1, 2)

N = 40
beta = 1.0
B = 2.0
mlE, _U, E0, _F = calc_energies(N = N, beta = beta, B = B)

print("Average energy:       " + "{:.5f}".format(_U))
print("Helmholtz energy:     " + "{:.5f}".format(_F))
print("Most likely energy:   " + "{:.3f}".format(mlE))
print("Smallest energy:      " + "{:.3f}".format(E0))

#
# Plot this
#
t = np.arange(0, 10)
y0 = [mlE for _t in t]
y1 = [_U for _t in t]
y2 = [E0 for _t in t]
y3 = [_F for _t in t]
ax0.plot(t, y0, label = "Most likely energy")
ax0.plot(t, y1, "--", label = "Average energy")
ax0.plot(t, y2, "k--", label = "Lowest energy")
ax0.plot(t, y3, ":", label = "Helmholtz energy")
ax0.legend()
ax0.tick_params(bottom = False, labelbottom = False)
#
# Plot this for various values of B
# 
_mlE = []
_avg = []
_min = []
_helmholtz = []
_B = np.arange(0.05, 1.0, 0.005)
for B in _B:
    mlE, _U, E0, _F = calc_energies(N = N, beta = beta, B = B)
    _mlE.append(mlE)
    _avg.append(_U)
    _min.append(E0)
    _helmholtz.append(_F)

ax1.plot(_B, _mlE, label = "Most likely energa")
ax1.plot(_B, _avg, "--", label = "Average energy")
ax1.plot(_B, _helmholtz, ":", label = "Helmholtz energy")
ax1.set_xlabel("Field strength B")
plt.savefig("TwoStateSystemEnergyLevels.png")
plt.show()