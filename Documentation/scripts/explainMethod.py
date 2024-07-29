import numpy as np
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MC2ISR_functions import *

# Make the delta v and velocity mesh
dv_order = 0.
dv = 10**dv_order
v = np.arange(-4,4+dv,dv)

z = math.pi/12 - 1e-3*1j

# Make a function for the Maxwellian
def f(v):
    return np.exp(-v*v)

# A highly refined velocity mesh for the exact Maxwellian
vFinePlot = np.linspace(-5,5,1001)

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

# Make styles
exactStyle = dict(color='C0',linestyle='-',label='Exact',linewidth=4)
poleStyle = dict(color='k',linestyle='--',linewidth=1.5, label='Re($z$)')
linearStyle = dict(color='C8',linestyle='-',linewidth=2)
linearbackgroundStyle = dict(color='k',linestyle='-',linewidth=2.7)
discreteStyle = dict(color='k',linestyle='none',marker='s',markersize=8,label='Discrete')
refinedStyle = dict(color='C3', marker='.',linestyle='none',markersize=11,label='Longley\n2024')

# Plot location of real component of pole
ax.axvline(x=np.real(z), **poleStyle)

# Plot exact Maxwellian
ax.plot(vFinePlot, f(vFinePlot),**exactStyle)

# Connect the dots and plot the the linearly interpolated distribution function
for i in range(len(a)):
    vCell = np.linspace(v[i], v[i+1], 101)
    ax.plot(vCell, a[i]*vCell+b[i],**linearbackgroundStyle)
    if i == 1:
        ax.plot(vCell, a[i]*vCell+b[i],**linearStyle,label='Linear\nInterp')
    else:
        ax.plot(vCell, a[i]*vCell+b[i],**linearStyle)
    

# Plot the pole refined points
ax.plot(vRefined, f_interp(vRefined),**refinedStyle)

# Plot the discrete points
ax.plot(v, f(v), **discreteStyle)

# Set up legend
handles, labels = ax.get_legend_handles_labels()
sortedIndex = [0,1,4,2,3]
labels = [labels[i] for i in sortedIndex]
handles = [handles[i] for i in sortedIndex]
ax.legend(handles, labels,loc='lower left',bbox_to_anchor=(-.01,0.33))

# Make plot look nice
ax.set_xlabel('$v_\parallel/v_{th}$')
ax.set_ylabel('$f_{0s}$')
ax.set_xlim(-4.2,4.2)
ax.set_ylim(-.02,1.02)
ax.grid()

fig.savefig('C:/Users/Chirag/Documents/2024_PoP_ISR/figures/explain.pdf',format='pdf')
    
    
    
    
    
    
    
    
    
    
    