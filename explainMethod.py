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

z = 0.25 - 1e-4*1j


def f(v):
    return np.exp(-v*v)

vFinePlot = np.linspace(-4,4,1001)

vRefined = getVInterp(np.array([z]), v, 500)
f_interp = interp1d(v, f(v))

[a,b] = getLinearInterpCoeff(v, f(v))


# Initialize figure
font = {'size'   : 22}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(8,6))
gs = gridspec.GridSpec(1,1)
gs.update(left=0.14, right=.99, bottom=.14, top=.95, wspace=0.15, hspace=.015)
fig.patch.set_facecolor('white')
ax = plt.subplot(gs[0])

ax.plot(vFinePlot, f(vFinePlot),'-',color='C0',linewidth=3,label='Exact')
ax.axvline(x=np.real(z), color='k',linestyle='--')

ax.set_xlim(-4.2,4.2)
ax.set_ylim(-.02,1.02)
ax.grid()
ax.set_xlabel('$v/v_{th}$')
ax.set_ylabel('$f$')
ax.legend()

fig.savefig('C:/Users/Chirag/Documents/repos/OO_Collisions/Documentation/figures/explain_1.pdf',format='pdf')

ax.plot(v, f(v), '.k',markersize=15,label='Discrete')
ax.legend()
fig.savefig('C:/Users/Chirag/Documents/repos/OO_Collisions/Documentation/figures/explain_2.pdf',format='pdf')

ax.plot(vRefined, f_interp(vRefined),'.',color='C3',markersize=8,label='Refined')
ax.legend()
fig.savefig('C:/Users/Chirag/Documents/repos/OO_Collisions/Documentation/figures/explain_3.pdf',format='pdf')

for i in range(len(a)):
    vCell = np.linspace(v[i], v[i+1], 101)
    if i == 1:
        ax.plot(vCell, a[i]*vCell+b[i],color='C1',label='Linear\nInterp')
    else:
        ax.plot(vCell, a[i]*vCell+b[i],color='C1',label='_nolegend_')
ax.legend()
    

fig.savefig('C:/Users/Chirag/Documents/repos/OO_Collisions/Documentation/figures/explain_4.pdf',format='pdf')
    
    
    
    
    
    
    
    
    
    
    