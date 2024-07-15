import numpy as np
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MC2ISR_functions import *

# Make the delta v and velocity mesh
dv_order = -0.
dv = 10**dv_order
v = np.arange(-4,4+dv,dv)

z = 0.25 - 1e-3*1j

# Make a function for the Maxwellian
def f(v):
    return np.exp(-v*v)

# A highly refined velocity mesh for the exact Maxwellian
vFinePlot = np.linspace(-4,4,1001)

# Make a refined velocity mesh based on Longley 2024 
vRefined = getVInterp(np.array([z]), v, 500) 
# We only want the values in between the known discrete points
# Since we are setting the pole at Re(z)=0.25, we only want he points between 0 and 1
vRefined = vRefined[vRefined > 0]
vRefined = vRefined[vRefined < 1]

# Get the linear interpolation function for the pole refinement
f_interp = interp1d(v, f(v))

# Get the linear interpolation coefficients
[a,b] = getLinearInterpCoeff(v, f(v))

# Initialize figure
font = {'size'   : 22}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(8,6))
gs = gridspec.GridSpec(1,1)
gs.update(left=0.14, right=.99, bottom=.14, top=.95, wspace=0.15, hspace=.015)
fig.patch.set_facecolor('white')
ax = plt.subplot(gs[0])

# Plot exact Maxwellian
ax.plot(vFinePlot, f(vFinePlot),'-',color='C0',linewidth=3,label='Exact')

# Plot location of real component of pole
ax.axvline(x=np.real(z), color='k',linestyle='--')

# Plot the discrete points
ax.plot(v, f(v), '.k',markersize=15,label='Discrete')

# Connect the dots and plot the the linearly interpolated distribution function
for i in range(len(a)):
    vCell = np.linspace(v[i], v[i+1], 101)
    if i == 1:
        ax.plot(vCell, a[i]*vCell+b[i],color='C1',label='Linear\nInterp',linewidth=2)
    else:
        ax.plot(vCell, a[i]*vCell+b[i],color='C1',label='_nolegend_',linewidth=2)

# Plot the pole refined points
ax.plot(vRefined, f_interp(vRefined),'.',color='C3',markersize=11,label='Longley\n2024')


# Make plot look nice
ax.set_xlabel('$v_\parallel/v_{th}$')
ax.set_ylabel('$f_{0s}$')
ax.legend()
ax.set_xlim(-4.2,4.2)
ax.set_ylim(-.02,1.02)
ax.grid()

fig.savefig('C:/Users/Chirag/Documents/repos/OO_Collisions/Documentation/figures/explain.pdf',format='pdf')
    
    
    
    
    
    
    
    
    
    
    