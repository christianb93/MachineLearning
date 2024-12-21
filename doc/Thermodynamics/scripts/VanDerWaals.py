#########################################
# Draw a few diagrams for a van der 
# Waal fluid
# We set k = 1
#########################################

import numpy as np
import matplotlib.pyplot as plt
import argparse

#
# Normalize temperature and volume, i.e. divide
# by the temperature respective volume at the
# critical point
# Recall that
# T_c = 8a / 27b
# V_c = 3 bN
#
def norm(T,V, N, a = 1, b = 1):
    return 27*b/(8*a) * T, V / (3*b*N) 

#
# Determine the pressure as function of
# temperature, volume and particle number
#
def P(T,V, N, a = 1, b = 1):
    return  N*T/(V-b*N) - a*N*N/V**2 


#
# Determine the reduced pressure in terms of
# reduced volume and reduced temperature
#
def Pb(Tb, Vb):
    return 8.0*Tb/(3.0*Vb-1) - 3.0/Vb**2

#
# Determine the part of the reduced Gibbs energy that does not
# depend on the temperature in terms of reduced volume and temperature
# i.e. this is the molar Gibbs energy divided by k_B T_c and dropping
# the terms that do not depend on V
#
def Gb(Tb, Vb):
    return - Tb * (np.log(3.0*Vb - 1.0) - 1.0/(3.0*Vb - 1)) - 9.0/(4.0*Vb)

#
# Return the reduced Helmholtz energy
#
def Fb(Tb, Vb):
    return - Tb * np.log(3.0*Vb - 1.0) - 9.0/(8.0*Vb)


#
# Given a certain combination of reduced
# temperature and pressure, determine the
# three possible values of the reduced volume
#
def solve_forVb(Pb, Tb):
    roots = np.roots([3*Pb, -Pb-8*Tb, 9,-3])
    if len(roots) != 3:
        print("Got problem with Pb=", Pb, "Tb = ", Tb)
        print("Only ", len(roots), "roots found")
    return np.sort(roots)


#
# Approximate derivatives as finite differences
# for given values of x and y 
# Here x and y are supposed to be arrays 
#
def compute_derivatives(x,y):
    results = []
    for i in range(len(x)-1):
        derivative = (y[i+1] - y[i]) / (x[i+1] - x[i])
        results.append(derivative)
    return results


#
# Given a reduced temperature T_b, find the
# coexistence state, i.e. the pressure for which
# the Gibbs energies of the liquid and gas phase 
# are equal. Thus we
# - search a certain range of pressure values P_b
# - for each range, we determine the corresponding values for the volume
# - for each of the volumes, we calculate the Gibbs energies
# - we stop if the sign of the difference between the Gibbs energies of
#   the gas phase and the liquid phase changes
#
def find_coexistence_state(_Tb = 0.85, step = 0.001):
    diff = 0.1
    for _Pb in np.arange(0.01, 1.0, step):
        _Vb1, _Vb2, _Vb3 = solve_forVb(Pb = _Pb,Tb = _Tb)
        _Gbvalues = Gb(Tb = _Tb, Vb = np.array([_Vb1, _Vb2, _Vb3]))
        # Here we assume that the middle value of the volume is the
        # one in the unstable region
        diff_new = _Gbvalues[0] - _Gbvalues[2]
        if ((diff_new > 0) and (diff <= 0)) or ((diff_new <= 0) and (diff > 0)):
            return _Pb
        diff = diff_new
    return 0

#
# Parse arguments
# 
parser = argparse.ArgumentParser()
parser.add_argument("--show", 
                    action = "store_true",
                    help ="Show generated images")
args=parser.parse_args()
show = args.show

a = 1.05
b = 0.1
N = 1
Tc = 8*a / (27*b)
print("Predicted for critical temperature: ", 8*a / (27*b))
print("Predicted for critical volume: ", 3*b)

#
# First we make a plot around the critical
# region and show the dependency of the 
# pressure as a function of the volume
#

fig = plt.figure(figsize = (10,8))
ax0 = fig.add_subplot(2, 2, 1)
ax0.set_xlabel("Volume V")
ax0.set_ylabel("Pressure P")


V = np.arange(0.19, 0.8, 0.001)
for T in np.arange(2.9, 3.4, 0.1):
    values = P(T,V,N, a = a, b = b)    
    ax0.plot(V,values, "b")


# 
# Now we turn this around for a temperature
# below the critical point
#
ax1 = fig.add_subplot(2, 2, 2)
T = 2.9
Tbar, _ = norm(T,1, N, a = a, b = b)
values = P(T,V,N, a = a, b = b)    
ax1.plot(values, V, "g")
ax1.set_xlabel("Pressure P")
ax1.set_ylabel("Volume V")


#
# We now compute the derivative of P wrt to V
#
d = compute_derivatives(V, values)
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(V[1:], d, "g")
ax2.plot(V[1:], np.zeros(len(V)-1), "y")
ax2.set_xlabel("Volume V")
ax2.set_ylabel("dP / dV")

#
# Mark the zeros in the picture above
#
for i in range(len(V)-2):
    if ((d[i] >= 0) and (d[i+1] < 0)) or ((d[i] < 0) and (d[i+1] >= 0)):
        ax1.plot(values[i+1], V[i+1], "rD")
        ax2.plot(V[i+1], 0, "rD")

plt.savefig("VanDerWaals.png")        

#
# Now we turn to the potentials. First we plot the 
# normalized Gibbs potential and the corresponding 
# part of the P-V curve
# 
fig = plt.figure(figsize=(10,8))
ax0 = fig.add_subplot(2,1,1)
ax0.set_xlim(0, 6)
ax0.set_xlabel("Normalized volume")
ax0.set_ylabel("Normalized Gibbs energy g")

ax1 = fig.add_subplot(2,1,2)
ax1.set_xlabel("Normalized volume")
ax1.set_ylabel("Normalized pressure")
ax1.set_xlim(0, 6)

Tb = 0.85
Vb = np.arange(0.53, 6, 0.001)
_Pb = Pb(Tb, Vb)
values = Gb(Tb, Vb)   
ax0.plot(Vb, values)
ax1.plot(Vb, _Pb)

#
# Now let us mark a few points for different values
# of P
#
lines = ["r--", "g--", "y--"]
markers = ["rD", "gD", "yD"]
text = [["A","B","C"], ["D", "E", "F"], ["G", "H", "I"]]
offset = 0.02
i = 0
for _Pb in [0.505, 0.4]:
    Vb1, Vb2, Vb3 = solve_forVb(Pb = _Pb,Tb = 0.85)
    Gbvalues = Gb(Tb = 0.85, Vb = np.array([Vb1, Vb2, Vb3]))
    line_y = np.full(600, _Pb)
    ax1.plot(np.arange(0, 6, 0.01), line_y, lines[i])
    ax1.plot(Vb1, _Pb, markers[i])
    ax1.text(Vb1, _Pb + offset, text[0][i])
    ax1.plot(Vb2, _Pb, markers[i])
    ax1.text(Vb2, _Pb + offset, text[1][i])
    ax1.plot(Vb3, _Pb, markers[i])
    ax1.text(Vb3, _Pb + offset, text[2][i])
    ax0.plot(np.array([Vb1, Vb2, Vb3]), Gbvalues, markers[i])
    ax0.text(Vb1 - 0.2, Gbvalues[0] - offset*0.5, text[0][i])
    ax0.text(Vb2, Gbvalues[1]+ offset, text[1][i])
    ax0.text(Vb3, Gbvalues[2] + offset, text[2][i])
    ax0.plot(np.arange(0, 6, 0.01), np.full(600, Gbvalues[2]), lines[i])
    i += 1

plt.savefig("VanDerWaalsMaxwell.png")



#
# Let us now turn to the Helmholtz free energy and do a plot
# compared to the reduced pressure
#
fig = plt.figure(figsize=(10,8))
ax0 = fig.add_subplot(2, 1, 1)
ax1 = fig.add_subplot(2, 1, 2)
ax0.set_xlabel("Normalized volume")
ax0.set_ylabel("Normalized Helmholtz energy f")
ax1.set_xlabel("Normalized volume")
ax1.set_ylabel("Normalized pressure")
ax1.set_xlim(0, 5)
ax1.set_ylim(-1.0 , 2.0)
ax0.set_xlim(0, 5)
Tb = 0.78
Vb = np.arange(0.42, 5, 0.001)
_Pb = Pb(Tb, Vb)
values = Fb(Tb, Vb)   
ax0.plot(Vb, values)
ax1.plot(Vb, _Pb)

# Let us now look at a certain pressure in more detail
# namely the coexistence pressure for the given temperature
_Pb = find_coexistence_state(_Tb = Tb)
offset = 0.1
i = 0
Vb1, Vb2, Vb3 = solve_forVb(Pb = _Pb, Tb = Tb)
# draw a horizontal red line at the pressure 
line_y = np.full(600, _Pb)
ax1.plot(np.arange(0, 6, 0.01), line_y, "r--")
# add markers at the volume of gas and liquid phase
ax1.plot(Vb1, _Pb, "rD")
ax1.text(Vb1, _Pb + offset, "A")
ax1.plot(Vb3, _Pb, "rD")
ax1.text(Vb3, _Pb + offset, "B")
# add the same markers in the upper diagram
Fbvalues = Fb(Tb = Tb, Vb = np.array([Vb1, Vb3]))
ax0.plot(np.array([Vb1,  Vb3]), Fbvalues + offset*0.2, "v")
ax0.text(Vb1, Fbvalues[0] + offset*0.8, "A")
ax0.text(Vb3, Fbvalues[1] + offset*0.8, "B")
# finally draw a line from Vb3 to Vb1
t = np.arange(0, 1.0, 0.001)
_v = t* Vb3 + (1.0 - t) * Vb1
_fb = t* Fbvalues[1] + (1.0 - t) * Fbvalues[0]
ax0.plot(_v, _fb, "g--")

plt.savefig("VanDerWaalsHelmholtz.png")


# 
# Now look for coexistence states
# 
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1, title="Coexistence states")
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

ax.set_xlabel("Normalized temperature")
ax.set_ylabel("Normalized pressure")
ax.set_xlim(0.3, 1.3)
ax.set_ylim(0.0, 1.3)

values = []
Tb = np.arange(0.3, 0.999, 0.005)
for _Tb in Tb:
    # print("_Tb =", _Tb)
    Pb = find_coexistence_state(_Tb = _Tb)
    values.append(Pb)    
    
ax.plot(Tb, values)
ax.plot(1.0, 1.0, "bo")
    
plt.savefig("VanDerWaalsPhaseDiagram.png")

if show:
    plt.show()
