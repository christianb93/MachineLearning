#########################################
# Draw a few diagrams for a van der 
# Waal fluid
# We set k = 1
#########################################

import numpy as np
import matplotlib.pyplot as plt

def norm(T,V, N, a=1, b=1):
    return 27*b/(8*a) * T, V / (2*b*N) 

def P(T,V, N, a=1, b=1):
    return  N*T/(V-b*N) - a*N*N/(V*V) 

def Pb(Tb, Vb, N, a=1, b=1):
    return 8.0*Tb/(3.0*Vb-1) - 3.0/(Vb*Vb)

def solve_forVb(Pb, Tb):
    roots = np.roots([3*Pb, -Pb-8*Tb, 9,-3])
    if len(roots) != 3:
        print("Got problem with Pb=", Pb, "Tb = ", Tb)
        print("Only ", len(roots), "roots found")
    return np.sort(roots)

def U(T,V,N, a=1, b=1):
    return 1.5*N*T - a*N*N/V

def A(T,V,N, a=1, b=1):
    return -N*T*np.log((V-b*N)/N) - 1.5*N*T*np.log(T) - a*N*N/V

def Gb(Tb,Vb):
    return - Tb * (np.log(3.0*Vb - 1.0) - 1.0/(3.0*Vb - 1)) - 9.0/(4.0*Vb)


def S(T,V,N, a=1, b=1):
    return 1.5*N + N*(np.log((V-b*N)/N) + 1.5*np.log(T))


def compute_derivatives(x,y):
    results = []
    for i in range(len(x)-1):
        derivative = (y[i+1] - y[i]) / (x[i+1] - x[i])
        results.append(derivative)
    return results


def find_coexistence_state(_Tb = 0.85):
    diff = 0.1
    for _Pb in np.arange(0.01, 1.0, 0.002):
        _Vb1, _Vb2, _Vb3 = solve_forVb(Pb=_Pb,Tb=_Tb)
        _Gbvalues = Gb(Tb=_Tb, Vb=np.array([_Vb1, _Vb2, _Vb3]))
        diff_new = _Gbvalues[0] - _Gbvalues[2]
        if ((diff_new > 0) and (diff <= 0)) or ((diff_new <= 0) and (diff > 0)):
            return _Pb
        diff = diff_new
    return 0



a = 1.05
b = 0.1
N = 1
Tc = 8*a / (27*b)
print("Predicted for critical temperature: ", 8*a / (27*b))
print("Predicted for critical volume: ", 3*b)

#
# First we make a plot around the critical
# region and show the dependency of the 
# pressure from the volume
#

fig = plt.figure(figsize=(10,8))
ax0 = fig.add_subplot(2,2,1)
ax0.set_xlabel("V")
ax0.set_ylabel("P")


V = np.arange(0.19, 0.8, 0.001)
for T in np.arange(2.9, 3.4, 0.1):
    values = P(T,V,N, a = a, b = b)    
    ax0.plot(V,values, "b")


# 
# Now we turn this around for a temperature
# below the critical point
#
ax1 = fig.add_subplot(2,2,2)
T = 2.9
Tbar, _ = norm(T,1, N, a=a, b=b)
values = P(T,V,N, a = a, b = b)    
ax1.plot(values, V, "g")
ax1.set_xlabel("P")
ax1.set_ylabel("V")


#
# We now compute the derivative of P wrt to V
#
d = compute_derivatives(V, values)
ax2 = fig.add_subplot(2,1,2)
ax2.plot(V[1:], d, "g")
ax2.plot(V[1:], np.zeros(len(V)-1), "y")

#
# Mark the zeros in the picture above
#
for i in range(len(V)-2):
    if ((d[i] >= 0) and (d[i+1] < 0)) or ((d[i] < 0) and (d[i+1] >= 0)):
        ax1.plot(values[i+1], V[i+1], "rD")
        ax2.plot(V[i+1], 0, "rD")

#plt.savefig("VanDerWaals.png")        

#
# Now we turn to the potentials. First we plot the 
# normalized Gibbs potential and the corresponding 
# part of the P-V curve
# 
fig = plt.figure(figsize=(10,8))
ax0 = fig.add_subplot(2,1,1, title="Gibbs energy")
ax0.set_xlim(0, 6)
ax1 = fig.add_subplot(2,1,2, title="Normalized pressure")
ax1.set_xlim(0, 6)
Tb = 0.85
Vb = np.arange(0.53, 6, 0.001)
Pb = Pb(Tb, Vb, N, a=a, b=b)
values = Gb(Tb,Vb)   
ax0.plot(Vb,values)
ax1.plot(Vb, Pb)

#
# Now let us mark a few points for different values
# of P
#
lines = ["r--", "g--", "y--"]
markers = ["rD", "gD", "yD"]
text = [["A","B","C"], ["D", "E", "F"], ["G", "H", "I"]]
offset = 0.02
i = 0
for Pb in [0.505, 0.4]:
    Vb1, Vb2, Vb3 = solve_forVb(Pb=Pb,Tb=0.85)
    Gbvalues = Gb(Tb=0.85, Vb=np.array([Vb1, Vb2, Vb3]))
    line_y = np.full(600, Pb)
    ax1.plot(np.arange(0, 6, 0.01), line_y, lines[i])
    ax1.plot(Vb1, Pb, markers[i])
    ax1.text(Vb1, Pb + offset, text[0][i])
    ax1.plot(Vb2, Pb, markers[i])
    ax1.text(Vb2, Pb + offset, text[1][i])
    ax1.plot(Vb3, Pb, markers[i])
    ax1.text(Vb3, Pb + offset, text[2][i])
    ax0.plot(np.array([Vb1, Vb2, Vb3]), Gbvalues, markers[i])
    ax0.text(Vb1 - 0.2, Gbvalues[0] - offset*0.5, text[0][i])
    # ax0.text(Vb2, Gbvalues[1]+ offset, text[1][i])
    ax0.text(Vb3, Gbvalues[2] + offset, text[2][i])
    ax0.plot(np.arange(0, 6, 0.01), np.full(600, Gbvalues[2]), lines[i])
    i += 1

plt.savefig("VanDerWaalsMaxwell.png")


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
Tb = np.arange(0.3, 0.999, 0.001)
for _Tb in Tb:
    # print("_Tb =", _Tb)
    Pb = find_coexistence_state(_Tb = _Tb)
    values.append(Pb)    
    
ax.plot(Tb, values)
ax.plot(1.0, 1.0, "bo")
    
plt.savefig("VanDerWaalsPhaseDiagram.png")

plt.show()

