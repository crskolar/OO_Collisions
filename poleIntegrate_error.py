import numpy as np
import math 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits import mplot3d
import scipy.special as sp
from scipy.interpolate import interp1d
from MC2ISR_functions import *
import time
import os

# Make functions for exact solutions
def p1(z):
    return np.exp(-z**2)*(-math.pi*sp.erfi(z)+np.log(-1/z)+np.log(z))
    # return -2*math.pi**.5*sp.dawsn(z)+np.sign(np.imag(z))*1j*math.pi*np.exp(-z**2)

def p2(z):
    return -2*math.pi**.5-2*z*p1(z)

def pstar(z):
    return 1j*(p1(np.conjugate(z))-p1(z))/2/np.imag(z)

def ones(z):
    return np.zeros_like(z)

def makePoles_p1(z):
    return [np.array([z]), np.array([1])]

def makePoles_p2(z):
    return [np.array([z]), np.array([2])]

def makePoles_pstar(z):
    return [np.array([z, np.conjugate(z)]), np.array([1,1])]

def calcIntegrand(f, v, z, order):
    integrand = f + 1j*0.0
    for i in range(len(z)):
        integrand /= (v-z[i])**order[i]
    return integrand

# Make the whole process a function so that we can just call it 3 times
# poleType determines which type of pole we are trying to test
# Is a string where the options are:
# p1
# p2
# pstar
def plotComparisons(poleType):  
    print(poleType)
    # Set figure stuff. Fix to make look nice later. Then copy it to do other integrals
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(17,10))
    gs = gridspec.GridSpec(5,3, height_ratios=[1,1,.23,1,1])
    gs.update(left=0.065, right=.995, bottom=.085, top=.95, wspace=0.04, hspace=.015)
    fig.patch.set_facecolor('white')
    ax = []
    for i in range(0,15):
        if i < 6 or i > 8:
            ax.append(plt.subplot(gs[i])) 
    
    # Make an array of gammas ranging from 1e-7 to 1e0
    gamma = np.logspace(-6,0,51)
    real_z = 0.25  # This is the real part of z. Is arbitrary. Possible that it is more important when z is closest to distribution function maximum?
    z = real_z - 1j*gamma

    if poleType == "p1": # Pole at z of order 1
        exactFun = p1
        poleFun = makePoles_p1
        ylims_real = [-.9,-.18]
        ylims_imag = [-7,0.2]
    elif poleType == "p2": # Pole at z or order 2
        exactFun = p2
        poleFun = makePoles_p2
        # ylims_real = [-3.5,1]
        # ylims_imag = [-.05,2]
        ylims_real = [-5.5,0]
        ylims_imag = [-1,1]
    elif poleType == "pstar": # Pole at z and z*. Each order 1
        exactFun = pstar
        poleFun = makePoles_pstar
        ylims_real = [-.1e6,3.6e6]
        ylims_imag = [-.05,.05]
    elif poleType == "p2_ones": # Pole at z with order 2 for a distribution function of ones
        exactFun = ones
        poleFun = makePoles_p2
        ylims_real = [-10,10]
        ylims_imag = [-10,10]
        
        
    # Calculate exact solution
    exactSol = exactFun(z)
    
    # Initialize pole integration values
    p_approx = np.zeros_like(gamma)+1j*0.0
    p_naive = np.zeros_like(gamma)+1j*0.0

    mesh_n = 500

    for k in range(12):
        # Plot exact solutions
        if k in [0,1,2,6,7,8]: # Do real values
            ax[k].plot(gamma, np.real(exactSol),'k-',linewidth=3)
            ax[k].set_ylim(ylims_real)
        else: # Do imaginary values
            ax[k].plot(gamma, np.imag(exactSol),'k-',linewidth=3)
            ax[k].set_ylim(ylims_imag)
        ax[k].set_xscale('log')
        ax[k].grid()
        ax[k].set_xticks(np.logspace(-6,0,4))
        ax[k].set_xlim(10**-6.3,10**.3)
        if k < 9:
            ax[k].set_xticklabels([])
        else:
            ax[k].set_xlabel('$(\\nu/k_{\parallel})/v_{th}$')
        if k/3 != round(k/3):
            ax[k].set_yticklabels([])
        # else:
            # ax[k].yaxis.set_major_formatter(FormatStrFormatter('%.1'))

    counter = 0
    # We want to iterate through dv of 10^-1 to 10^-6
    for i in range(-1,-7,-1):
        dv = 10.**i
        
        # Build a velocity mesh going from -4+u to 4+u (this generally captures entire maxwellian)
        v = np.arange(-4, 4+dv,dv)
        
        # Make distribution function
        if poleType == "p2_ones":
            f0 = 0.*v + 1.0
        else:
            f0 = np.exp(-v**2)
        
        # Iterate through gamma
        print("dv:", dv)
        
        for k in range(len(gamma)): 
            # Calculate poles and orders
            [poles, orders] = poleFun(z[k])
            p_approx[k] = poleIntegrate(poles, orders, v, f0, mesh_n, 0)
            p_naive[k] = np.trapz(calcIntegrand(f0, v, poles, orders), v)
            
        ax[counter + int(counter/3)*3].set_title('$\Delta v/v_{th}=10^{%d}$' % (np.log10(dv)))    
        
        ax[counter + int(counter/3)*3].plot(gamma,np.real(p_naive),'.',color='C1')
        ax[counter + int(counter/3+1)*3].plot(gamma,np.imag(p_naive),'.',color='C1')
        
        ax[counter + int(counter/3)*3].plot(gamma,np.real(p_approx),'o',fillstyle='none',markersize=8,color='C0')
        ax[counter + int(counter/3+1)*3].plot(gamma,np.imag(p_approx),'o',fillstyle='none',markersize=8,color='C0')
        
        counter += 1
        
    ax[0].set_ylabel('Real')
    ax[6].set_ylabel('Real')
    ax[3].set_ylabel('Imag')
    ax[9].set_ylabel('Imag')
    
    ax[0].legend(['Exact','Trapz','Pole Refined'])
    
    print("Done!")
    
    fig.savefig('Documentation/figures/'+poleType+'.pdf',format='pdf')

plotComparisons("p1")
# plotComparisons("p2")
# plotComparisons("pstar")
# plotComparisons("p2_ones")
    