import numpy as np
import math
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import scipy.special as sp

import scipy.constants as const
u = const.physical_constants['unified atomic mass unit'][0]
kB = const.k
me = const.m_e
e = const.e
eps0 = const.epsilon_0
c = const.c

# Set background parameters based on a reasonable F region plasma 
B = 1e-5

Tiperp = 2000
Tipar = 1000
Dstar = 1.8
mi = 16*u
vthpar = (2*kB*Tipar/mi)**.5
vthperp = (2*kB*Tiperp/mi)**.5


vpar = np.linspace(-4*vthpar, 4*vthpar, 1000)
vperp = np.linspace(-4*vthperp, 4*vthperp, 1000)

[VVperp, VVpar] = np.meshgrid(vperp, vpar)

# Build distribution function
def toroidal_norm(vperp, vpar, vthperp, vthpar, Dstar):
    Cperp = vperp/vthperp
    Cpar = vpar/vthpar
    return sp.iv(0,2*Dstar*Cperp)*np.exp(-Cpar**2-Cperp**2-Dstar**2)/vthperp**2/vthpar/math.pi**1.5


f = toroidal_norm(VVperp, VVpar, vthperp, vthpar, Dstar)

font = {'size'   : 26}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(7,7))
gs = gridspec.GridSpec(1,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
gs.update(left=0.17, right=.78, bottom=.0, top=1, wspace=0.015, hspace=.015)
fig.patch.set_facecolor('white')
ax = plt.subplot(gs[0])

im = ax.pcolormesh(VVperp/vthperp, VVpar/vthpar, f,cmap='inferno')
ax.grid(alpha=0.3)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('$v_\\perp/v_{th,\\perp}$')
ax.set_ylabel('$v_\\parallel/v_{th,\\parallel}$')

# Adding the colorbar
cbaxes = fig.add_axes([.8, 0.193 , 0.03, 0.612])
cb = fig.colorbar(im, ax=ax, cax=cbaxes,label='$f_0$')

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

theta = np.deg2rad(10)
t = np.linspace(-10, 10,10)
vperp_line = t*np.sin(theta)
vpar_line = t*np.cos(theta)

ax.plot(vperp_line, vpar_line, linewidth=3,color='k')
ax.plot(vperp_line, vpar_line, '--', linewidth=2,color='w')

fig.savefig('C:/Users/Chirag/Documents/Conferences/CEDAR_2024/poster/toroidal_dist.pdf',format='pdf')
