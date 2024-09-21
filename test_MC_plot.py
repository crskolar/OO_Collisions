import numpy as np
from loadMCData import readDMP, readDMPV
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec





fileDir = 'C:/Users/Chirag/Documents/O+O/Monte-Carlo/'
# fileName = 'noE                 .DMPV'
# fileNameDMP = 'noE_v2              .DMP'
fileNameDMP = 'noE                 .DMP'
fileName = 'noE_v2              .DMPV'
# fileName2 = '1-E=0-O+O           .DMPV' 


[VVperp, VVpar, f0, extra] = readDMPV(fileDir + fileName)

[f0DMP, vperp, vpar, VVperpDMP, VVparDMP, f0Full] = readDMP(fileDir+fileNameDMP)
# dv = 1.7e3
# vperp = np.arange(0,24*dv,dv)
# vpar = np.arange(-23*dv,24*dv,dv)
# [VVperpDMP,VVparDMP] = np.meshgrid(vperp, vpar)

# f0Full = np.zeros((47,24))
# f0Full[23:,:] = f0DMP
# f0Full[0:23,:] = np.flip(f0DMP[1:,:],0)



fig = plt.figure(1, figsize=(10,6))
font = {'size'   : 22}
mpl.rc('font', **font)
gs = gridspec.GridSpec(1,2)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
gs.update(left=0.095, right=.99, bottom=0.3, top=.88, wspace=0.015, hspace=0.3)
ax = []
for i in range(0,2):
    ax.append(plt.subplot(gs[i]))
fig.patch.set_facecolor('white')

# ax[0].pcolormesh(VVperp, VVpar, extra,cmap='inferno')
ax[0].pcolormesh(VVperpDMP, VVparDMP,f0Full,cmap='inferno')
# ax[1].pcolormesh(VVperp, VVpar, f0,cmap='inferno')
ax[1].pcolormesh(f0DMP,cmap='inferno')
ax[0].set_ylabel('$v_\parallel$')
ax[1].set_xlabel('$v_\perp$')
ax[0].set_xlabel('$v_\perp$')