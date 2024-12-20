################################################
# Some visualizations of the microcanonical 
# distribution
# Make sure to run
# pip install ptable
################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import prettytable as pt

#
# We start with a two-state system and plot
# the entropy as a function of the energy
#
def S_twoState(U, N, epsilon):
    return (U/epsilon - N) * np.log(1 - U/(epsilon*N)) - (U/epsilon) * np.log(U/(epsilon*N))
            

N = 1000
epsilon = 1
U = np.arange(1, N*epsilon, epsilon)
S = S_twoState(U,N,epsilon)
D = []
for i in range(len(U)-1):
    D.append(epsilon/(S[i+1] - S[i]))


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(2,1,1)
ax.set_xlabel("")
ax.set_ylabel("S")
ax.plot(U,S)

ax = fig.add_subplot(2,1,2)
ax.plot(U[1:], D, "y")
ax.set_xlabel("U")
ax.set_ylabel("T")
plt.savefig("TwoStateSystem.png")


#
# Next we do the same for an Einstein solid. We use the
# exact formula first, without a Stirling approximation
#
def Omega_EinsteinSolid(N,q):
    return scipy.special.comb(q+N-1, q)

def S_EinsteinSolidExact(N,q):
    return np.log(Omega_EinsteinSolid(N,q))

def S_EinsteinSolidApprox(N,q):
    return (q+N-1)*np.log(1+q/N) - q*np.log(q/N)

def T_EinsteinSolidApprox(N,q):
    return 1 / np.log(N/q + 1)


N = 50
U = np.arange(1, 5*N, 1)
Omega = Omega_EinsteinSolid(N,U)
S = S_EinsteinSolidExact(N,U)
table = pt.PrettyTable()
#
# Compute temperatures
#
T = []
for i in range(len(U)-1):
    T.append(1/(S[i+1] - S[i]))

table.field_names=["U", "W", "S", "T"]
for i in range(5):
    table.add_row([U[i], Omega[i], S[i], T[i]])
print(table)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(2,1,1)
ax.set_xlabel("")
ax.set_ylabel("S")
ax.plot(U,S, "b")


ax = fig.add_subplot(2,1,2)
ax.plot(U[1:], T, "y")
ax.set_xlabel("U")
ax.set_ylabel("T")



#
# Now we do the same thing for a larger system using
# the approximations
# 
N = 500
U = np.arange(1, 50*N, 1)
S = S_EinsteinSolidApprox(N,U)
T = T_EinsteinSolidApprox(N,U)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(2,1,1)
ax.set_xlabel("")
ax.set_ylabel("S")
ax.plot(U,S, "b")


ax = fig.add_subplot(2,1,2)
ax.plot(U, T, "y")
ax.set_xlabel("U")
ax.set_ylabel("T")
plt.savefig("EinsteinSolid.png")



#
# Now we consider a combined system with Na and Nb 
# particles and total energy q = qa + qb
# Let us compute the number Wa and Wb of microstates
# that each subsystem can have so that the total 
# energy is still q
# We reconstruct the tables and pictures in section 2.3
# of Schroeder, An introduction to thermal physics
#
Na = 1
Nb = 12
q = 20


table = pt.PrettyTable()
table.field_names = ["qa", "Wa", "qb", "Wb", "W", "CDF", "S"]
#
# For each possible value qa = 0, .., q
# let us calculate Wa and Wb
#
Wvalues = []
qavalues = []
cdf = []
Svalues = []
cumulated = 0
for qa in range(q+1):
    qb = q - qa
    Wa = Omega_EinsteinSolid(Na, qa)
    Wb = Omega_EinsteinSolid(Nb, qb)
    S = np.log(Wa*Wb)
    cumulated += Wa*Wb
    table.add_row([qa, Wa, qb, Wb, Wa*Wb, cumulated, S])
    qavalues.append(qa)
    Wvalues.append(Wa*Wb)
    cdf.append(cumulated)
    Svalues.append(S)

print(table)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-1,15)
ax.bar(qavalues, Wvalues / cumulated)
#
# Plot the canonical Boltzmann distribution as comparison
#
table = pt.PrettyTable()
table.field_names = ["qa", "p", "predicted"]
Ttot = T_EinsteinSolidApprox(Na+Nb,q)
beta = 1 / Ttot
predict = []
Z = 0
for i in range(len(qavalues)):
    p = np.exp((-1.0)*qavalues[i]*beta)
    Z += p
    predict.append(p)
ax.plot(qavalues, predict / Z, "y")
ax.set_xlabel("u(s)")
ax.set_ylabel("p(s)")
plt.savefig("BoltzmannEinsteinSolid.png")
U = 0
for i in range(len(qavalues)):
    p = Wvalues[i] / cumulated
    U += qavalues[i]*p
    table.add_row([qavalues[i], Wvalues[i] / cumulated, predict[i] / Z])
print(table)
print("Average energy U: ", U)
print("Temperature: ", Ttot)
print("Beta: ", beta)