# This script plots some kappa distributions so that we can understand what they look like

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import scipy.special as sp
from ISRSpectraFunctions import *

# Load constants
import scipy.constants as const
u = const.physical_constants['unified atomic mass unit'][0]
kB = const.k
mi = 16*u
    
# Set distribution parameters
T = 3000.

# Make an array of kappas to use
p = np.array([2,3,4,5,89])


# Make an array for vpar
vth = (2*kB*T/mi)**.5
v = np.linspace(-8*vth,8*vth,1000)

# Initialize figure
font = {'size'   : 22}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(8,6))
gs = gridspec.GridSpec(1,1)
gs.update(left=0.14, right=.99, bottom=.14, top=.95, wspace=0.15, hspace=.015)
fig.patch.set_facecolor('white')    
ax = plt.subplot(gs[0])

fMaxwell = maxwellian_norm(0.0, v, vth)
ax.plot(v/vth, fMaxwell, linewidth=4,color='k')

# Iterate through the kappas
for i in range(len(p)):
    # Calculate wk
    
    # wk = ((2*kappa[i]-3)*kB*T/kappa[i]/mi)**.5
    f = superG_norm(0.0, v, vth, p[i])
    
    ax.plot(v/vth, f,'--',linewidth=2)
    # ax.set_yscale('log')
    # ax.set_ylim(1e-4,1)
    ax.set_xlim(-8,8)
ax.grid()