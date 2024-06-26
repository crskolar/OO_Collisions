import numpy as np
import math 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits import mplot3d
import scipy.special as sp
from scipy.interpolate import interp1d
from MC2ISR_functions import *
import time
import os

dataDir = 'C:/Users/Chirag/Documents/O+O/O+O_ions_correct_colls/'

# Load the data
U_i = np.loadtxt(dataDir + 'U.txt', dtype=np.complex_)
M_i = np.loadtxt(dataDir + 'M.txt', dtype=np.complex_)
chi_i = np.loadtxt(dataDir + 'chi.txt', dtype=np.complex_)
U_e_exact = np.loadtxt(dataDir + 'U_e_exact.txt', dtype=np.complex_)
M_e_exact = np.loadtxt(dataDir + 'M_e_exact.txt', dtype=np.complex_)
chi_e_exact = np.loadtxt(dataDir + 'chi_e_exact.txt', dtype=np.complex_)
U_i_exact = np.loadtxt(dataDir + 'U_i_exact.txt', dtype=np.complex_)
M_i_exact = np.loadtxt(dataDir + 'M_i_exact.txt', dtype=np.complex_)
chi_i_exact = np.loadtxt(dataDir + 'chi_i_exact.txt', dtype=np.complex_)
S_exact = np.loadtxt(dataDir + 'S_exact.txt', dtype=np.complex_)
omega = np.loadtxt(dataDir + 'omega.txt')

# Get the last n for which we have data
for n in range(len(U_i[0])):
    if U_i[0,n] == 0:
        nEnd = n-1
        break
nEnd = 0

# Calculate ISR spectra
S = np.transpose(calcSpectra(np.transpose(M_i), M_e_exact, np.transpose(chi_i), chi_e_exact))

# Set colors based on length of phi_bias using inferno colormap\n",
colors = []
for i in range(0,nEnd+1):
    colors.append(pl.cm.inferno( i/(nEnd+1)) )
    
font = {'size'   : 22}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(10,12))
gs = gridspec.GridSpec(6,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
gs.update(left=0.3, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
fig.patch.set_facecolor('white')
ax = []
for i in range(0,6):    
    ax.append(plt.subplot(gs[i])) 

# Set dummy plot to create mappable surface for color bar\n",
c = np.arange(0., nEnd,10)
cmap = plt.get_cmap("inferno", len(c))
sm = plt.cm.ScalarMappable(cmap=cmap)
c = np.linspace(0, nEnd,num=nEnd+1)
#dummie_cax = ax[0].scatter(c, c, c=c, cmap=cmap)
#ax[0].cla()


for n in range(nEnd,nEnd+1):
    ax[0].plot(omega,np.real(U_i[:,n]))#,color=colors[n])
    ax[1].plot(omega,np.imag(U_i[:,n]))#,color=colors[n])
    ax[2].plot(omega,np.real(M_i[:,n]))#,color=colors[n])
    ax[3].plot(omega,np.real(chi_i[:,n]))#,color=colors[n])
    ax[4].plot(omega,np.imag(chi_i[:,n]))#,color=colors[n])
    ax[5].plot(omega,np.imag(S[:,n]))#,color=colors[n])
#     ax[1].plot(omega,np.imag(U_i[:,n]),color=colors[n])
#     ax[2].plot(omega,np.real(M_i[:,n]),color=colors[n])
#     ax[3].plot(omega,np.imag(M_i[:,n]),color=colors[n])
#     ax[4].plot(omega,np.real(chi_i[:,n]),color=colors[n])
#     ax[5].plot(omega,np.imag(chi_i[:,n]),color=colors[n])
    
#     # ax[0].plot(omega,U_i[:,n],color=colors[n])

ax[0].plot(omega,np.real(U_i_exact),'k--',linewidth=3)
ax[1].plot(omega,np.imag(U_i_exact),'k--',linewidth=3)
ax[2].plot(omega,np.real(M_i_exact),'k--',linewidth=3)
ax[3].plot(omega,np.real(chi_i_exact),'k--',linewidth=3)
ax[4].plot(omega,np.imag(chi_i_exact),'k--',linewidth=3)
ax[5].plot(omega,S_exact,'k--',linewidth=3)
# ax[6].plot(omega,np.real(U_e))
# ax[7].plot(omega,np.imag(U_e))
# ax[8].plot(omega,np.real(M_e))
# ax[9].plot(omega,np.imag(M_e))
# ax[10].plot(omega,np.real(chi_e))
# ax[11].plot(omega,np.imag(chi_e))
# ax[12].plot(omega,S[:,nEnd])

# ax[0].set_ylabel('Re$(U_i$)')
# ax[1].set_ylabel('Im$(U_i$)')
# ax[2].set_ylabel('Re$(M_i$)')
# ax[3].set_ylabel('Im$(M_i$)')
# ax[4].set_ylabel('Re$(\chi_i$)')
# ax[5].set_ylabel('Im$(\chi_i$)')
# ax[6].set_ylabel('Re$(U_e$)')
# ax[7].set_ylabel('Im$(U_e$)')
# ax[8].set_ylabel('Re$(M_e$)')
# ax[9].set_ylabel('Im$(M_e$)')
# ax[10].set_ylabel('Re$(\chi_e$)')
# ax[11].set_ylabel('Im$(\chi_e$)')
# ax[12].set_ylabel('S')
ax[5].set_xlabel('$\omega$ (rad/s)')

for k in range(6):
    ax[k].grid()



cbar_ax = fig.add_axes([.97, 0.62, .03, .3])
# cbar = fig.colorbar(sm,cax=cbar_ax,label='$n$',ticks=c,orientation='vertical')







