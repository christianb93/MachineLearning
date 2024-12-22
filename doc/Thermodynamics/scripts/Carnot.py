##################################
# Create a Carnot cycle diagram
##################################

import numpy as np
import matplotlib.pyplot as plt
import argparse

#
# Global variables
#
c = 1.5
gamma = 1 + 1.0 / c
d = 0.005 # stepsize
N = 1.0
R = 1.0

#
# Calculate the temperature and entropy for a given pressure
# and volume
# 
def entropy(P, V):
    #
    # First determine the temperature
    # PV = N R T
    #
    T = P * V / ( N * R)
    #
    # now use the standard expression for the entropy in terms of 
    # temperature and volume
    #
    S = c * N * R * np.log(T) + N * R * np.log(V)
    return T, S


#
# Return an array with P-values and an 
# array with V-values that model an isotherm
# connecting V0 and V1. Assume V1 > V0
#
def create_isotherm(P0, V0, V1):
    pValues = []
    vValues = []
    tValues = []
    sValues = []
    c = 1.0 * P0 * V0
    for V in np.arange(V0, V1, d):
        # PV  = c
        P = (1.0 * c) / (1.0 * V)
        t, s = entropy(P, V)
        vValues.append(V)
        pValues.append(P)
        tValues.append(t)
        sValues.append(s)
    return pValues, vValues, tValues, sValues

#
# Return an array with P-values and an 
# array with V-values that model an adiabat
# connecting V0 and V1. Assume V1 > V0
#
def create_adiabat(P0, V0, V1):
    pValues = []
    vValues = []
    tValues = []
    sValues = []
    c = 1.0 * P0 * V0**gamma
    for V in np.arange(V0, V1, d):
        # PV**gamma  = c
        P = (1.0 * c) / (1.0 * V**gamma)
        vValues.append(V)
        pValues.append(P)
        t, s = entropy(P, V)
        tValues.append(t)
        sValues.append(s)
    return pValues, vValues, tValues, sValues

#
# Parse arguments
# 
parser = argparse.ArgumentParser()
parser.add_argument("--show", 
                    action = "store_true",
                    help ="Show generated images")
args=parser.parse_args()
show = args.show


#
# Plot a Carnot cycle in P-V space
#

#
# Starting point
# 
V0 = 2.0
P0 = 2.0
#
# Volume after end of isothermal phase
#
V1 = 4.2
#
# Volume after end of adiabatic expansion phase
#
V2 = 6.0

fig = plt.figure(figsize = (10, 10))
ax0 = fig.add_subplot(2, 1, 1)
ax1 = fig.add_subplot(2, 1, 2)
ax0.set_xlabel("Volume V")
ax0.set_ylabel("Pressure P")
ax1.set_xlabel("Entropy S")
ax1.set_ylabel("Temperature T")


#
#  do isotherm from V0 to V1
#
pValues, vValues, tValues, sValues = create_isotherm(P0, V0, V1)
P1 = pValues[-1]
T0 = tValues[0]
S0 = sValues[0]
T1 = tValues[-1]
S1 = sValues[-1]

ax0.plot(vValues, pValues, "b")
ax1.plot(sValues, tValues, "b")

ax0.plot(V0, P0, "D")
ax0.text(V0 + 0.1, P0, "I")
ax0.plot(V1, P1, "D")
ax0.text(V1 + 0.1, P1 , "II")

ax1.plot(S0, T0, "D")
ax1.text(S0 + 0.01, T0 - 0.06, "I")
ax1.plot(S1, T1, "D")
ax1.text(S1 + 0.01, T1 - 0.06, "II")

#
#  do adiabat from V1 to V2
#
pValues, vValues, tValues, sValues = create_adiabat(P1, V1, V2)
ax0.plot(vValues, pValues, "b")
ax1.plot(sValues, tValues, "b")
P2 = pValues[-1]
S2 = sValues[-1]
T2 = tValues[-1]

ax0.plot(V2, P2, "D")
ax0.text(V2, P2 + 0.05 , "III")

ax1.plot(S2, T2, "D")
ax1.text(S2 + 0.01, T2 + 0.03 , "III")


#
# Next we want an isotherm taking us back to a lower volume V3. For the volume V3, we have two conditions
# first, (P3, V3) and (P2, V2) must be on the same isotherm, i.e
# P3 * V3 = P2 * V2
# and second, (P3, V3) and (P0, V0) must be on the same adiabat, i.e.
# P3 * V3**gamma = P0 * V0**gamma
#
# Dividing the second by the first relation yields
#
# V3**(gamma - 1.0) = (P0 * V0**gamma) / (P2 * V2)
#
# and we can then find P3 as
# P3 = P2 * V2 / V3  
# 
V3 = ((P0 * V0**gamma) / (P2 * V2))**(1.0 / (gamma - 1.0))
P3 = P2 * V2 / V3  

pValues, vValues, tValues, sValues = create_isotherm(P3, V3, V2)
S3 = sValues[0]
T3 = tValues[0]
ax0.plot(vValues, pValues, "b")
ax1.plot(sValues, tValues, "b")
ax0.plot(V3, P3, "D")
ax0.text(V3 + 0.1, P3 , "IV")
ax1.plot(S3, T3, "D")
ax1.text(S3 + 0.01, T3 + 0.03 , "IV")

#
# finally do an adiabat back to the starting point
#
pValues, vValues, tValues, sValues = create_adiabat(P0, V0, V3)
ax0.plot(vValues, pValues, "b")
ax1.plot(sValues, tValues, "b")


#
# and save
# 
plt.savefig("CarnotCycle.png")

#
# Now do the same in S - T space
#

if args.show:
    plt.show()

