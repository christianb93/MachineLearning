#############################################
# Planck law for a black body
#############################################


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def f(x):
    return x**3 / (np.exp(x) - 1)

def g(x):
    return 3*np.exp(x) - x *np.exp(x) - 3

#
# Determine maximum
#
m = scipy.optimize.newton(g, x0 = 2.6)
print("Maximum at: ", m)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)

ax.set_xlabel("Photon energy/ kT")
ax.set_ylabel("Energy density")

x = np.arange(0.1, 15, 0.1)
ax.plot(x, f(x))

plt.savefig("PlanckSpectrum.png")

