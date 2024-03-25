import numpy as np
import math 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.special as sp
from scipy.interpolate import interp1d
from MC2ISR_functions import *
import time

# Set physical constants
import scipy.constants as const
u = const.physical_constants['unified atomic mass unit'][0]
kB = const.k
me = const.m_e
e = const.e
eps0 = const.epsilon_0
c = const.c

# Update these test poleIntegrate functions to account for inclusion of int_dir
# Probably good to also have a test for 2D case as well. To show that it works properly

# This is the exact solution for an integral of a Guassian with a single pole
def singlePoleExact(z):
    return -2*math.pi**.5*sp.dawsn(z)+np.exp(-z*z)*np.log(-1/z)+np.exp(-z*z)*np.log(z)
    

# Tested
def test_getVInterp():
    # Choose two poles
    poles = np.array([2+1j, 3-2j])
    mesh_n = 11
    
    # Make an initial velocity grid
    v = np.arange(0,11,1)
    
    v_int = getVInterp(poles, v, mesh_n)
    
    # Hand calculation for these exact vs is (See my notes for this)
    v_exact = np.array([0,.4,.6,.8,1,1.2,1.4,1.6,2,2.2,2.4,2.8,3,3.2,3.6,3.8,4,4.6,4.8,5,5.4,5.6,6,6.2,6.4,7,7.2,8,8.6,8.8,9,9.6,10])
    
    # If these are close to 0, then this function is working correctly
    print(v_int - v_exact)
    return

# Tested.
def test_getIntegrand():
    v = np.array([0,1,2])
    Fv = np.array([2,3,1])
    poles = np.array([2+1j, 3-2*1j])
    orders = np.array([1,2])
    
    # Calculate integrand
    integrand = getIntegrand(poles, orders, v, Fv)
    integrand_exact = np.array([-0.05207100591715976331360946745562130177514792899408284023668639 -0.04497041420118343195266272189349112426035502958579881656804733*1j
                                ,-0.1875-0.1875*1j,-0.16-0.12*1j]) # From wolfram alpha
    print(integrand)    
    print(integrand_exact)
    print(integrand-integrand_exact)
    
    return

# Tested
def test_plemelj():
    v = np.array([1,2,3])
    Fv = np.array([1,1,1])# np.exp(-(v-1)**2)
    poles = np.array([2-1j,3+2j,1-5*1j])
    orders = np.array([1,2,1])
    
    exact_sol = 1j*math.pi*(- (11+7*1j)*Fv[1]/170 + (96+247*1j)*Fv[2]/140450 + (26+15*1j)*Fv[0]/901)
    
    plemelj_calc = plemelj(poles, orders, v, Fv)
    
    print(exact_sol)
    print(plemelj_calc)  
    return
    
# This function will test poleIntegrate in several ways.
# Get rid of this, we don't need it anymore.
# First, we will show that the Plemelj theorem only works for poles close to the real axis (i.e. is for limit as pole approaches real axis)
def test_poleIntegrate_ones_vs_Plemelj_changeGamma():
    
    # Show that pole needs to be close to real axis to get good result
    mesh_n = 100
    
    numPoints = 200
    gamma = np.logspace(-5,2,numPoints)
    approx_calc = gamma*0.0 + 1j*0.0
    plemelj_calc = gamma*0.0 +1j*0.e0
    
    orders = np.array([1])
    for k in range(numPoints):
        dv = 0.5*gamma[k]
        if dv > 0.01:
            dv = 0.01
        v = np.arange(-10, 20+dv, dv)
        Fv = np.ones_like(v)
        poles = np.array([2 - 1j*gamma[k]])
        approx_calc[k] = poleIntegrate(poles, orders, v, Fv, mesh_n, 0)
        plemelj_calc[k] =  plemelj(poles, orders, v, Fv)
    
    im_error = np.abs(np.imag(plemelj_calc - approx_calc) / np.imag(plemelj_calc))
    
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(7,6))
    gs = gridspec.GridSpec(2,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    gs.update(left=0.2, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    fig.patch.set_facecolor('white')
    ax = []
    for i in range(0,2):
        ax.append(plt.subplot(gs[i])) 
    
    ax[0].plot(gamma, np.real(approx_calc))
    ax[1].plot(gamma, np.imag(approx_calc),linewidth=2)
    ax[1].plot(gamma, np.imag(plemelj_calc),'k--',linewidth=2)
    ax[1].set_xlabel('$\gamma$')
    ax[0].set_ylabel('Real')
    ax[1].set_ylabel('Imag')
    ax[1].legend(['Approx', 'Exact'])
    
    for k in range(2):
        ax[k].set_xscale('log')
        ax[k].set_xlim(np.min(gamma), np.max(gamma))
        ax[k].set_xticks([1e-5,1e-3,1e-1,1e1])
        ax[k].grid()
    ax[0].set_xticklabels([])
    
    fig.savefig('Documentation/figures/ones_vs_Plemelj_changeGamma.png',format='png')
    

def test_poleIntegrate_Gaussian_vs_Plemelj_changeGamma():
    # Show that pole needs to be close to real axis to get good result
    mesh_n = 100
    
    numPoints = 50
    
    gamma = np.logspace(-5,.76,numPoints)
    approx_calc = gamma*0.0 + 1j*0.0
    plemelj_calc = gamma*0.0 +1j*0.0
    exact_solution = gamma*0.0 +1j*0.0
    naive_calc = gamma*0.0 + 1j*0.0
    
    orders = np.array([1])
    for k in range(numPoints):
        dv = 1e-5
        v = np.arange(-10, 20+dv, dv)
        Fv = np.exp(-v**2)
        poles = np.array([2 - 1j*gamma[k]])
        approx_calc[k] = poleIntegrate(poles, orders, v, Fv, mesh_n, 0)
        plemelj_calc[k] =  plemelj(poles, orders, v, Fv)
        naive_calc[k] = np.trapz(Fv/(v-poles[0]), v)
        exact_solution[k] = singlePoleExact(poles[0])
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(7,6))
    gs = gridspec.GridSpec(2,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    gs.update(left=0.23, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    fig.patch.set_facecolor('white')
    ax = []
    for i in range(0,2):
        ax.append(plt.subplot(gs[i])) 
    
    ax[0].plot(gamma, np.real(approx_calc),linewidth=2)
    ax[0].plot(gamma, np.real(naive_calc),'.')
    ax[0].plot(gamma, np.real(exact_solution),'--',linewidth=2,color='C3')
    # ax[0].set_ylim(-1.5,0.5)
    ax[1].plot(gamma, np.imag(approx_calc),linewidth=2)
    ax[1].plot(gamma, np.imag(naive_calc),'.')
    ax[1].plot(gamma, np.imag(exact_solution),'--',linewidth=2,color='C3')
    ax[1].plot(gamma, np.imag(plemelj_calc),'k--',linewidth=2)
    ax[1].set_xlabel('$\gamma$')
    ax[0].set_ylabel('Real')
    ax[1].set_ylabel('Imag')
    ax[1].legend(['Pole Refined', 'Naive', 'Exact', 'Plemelj'],loc='lower left',bbox_to_anchor=(.1,1.2,3,4))
    
    for k in range(2):
        ax[k].set_xscale('log')
        ax[k].set_xlim(np.min(gamma), np.max(gamma))
        ax[k].set_xticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
        ax[k].grid()
    ax[0].set_xticklabels([])
    ax[1].set_ylim(-1.3,.15)

    fig.savefig('Documentation/figures/Gaussian_vs_Plemelj_changeGamma_1e-5.png',format='png')


def test_poleIntegrate_Gaussian_vs_Plemelj_changeGamma_doublePole():
    # Show that pole needs to be close to real axis to get good result
    mesh_n = 100
    
    numPoints = 50
    gamma = np.logspace(-5,.76,numPoints)
    # poles_array = 
    approx_calc = gamma*0.0 + 1j*0.0
    plemelj_calc = gamma*0.0 +1j*0.0
    exact_solution = gamma*0.0 +1j*0.0
    naive_calc = gamma*0.0 + 1j*0.0
    
    orders = np.array([1,1])
    for k in range(numPoints):
        dv = 1e-3
        v = np.arange(-10, 20+dv, dv)
        Fv = np.exp(-(v)**2)
        poles = np.array([2 - 1j*gamma[k], 2+1j*gamma[k]])
        approx_calc[k] = poleIntegrate(poles, orders, v, Fv, mesh_n, 0)
        plemelj_calc[k] =  plemelj(poles, orders, v, Fv)
        naive_calc[k] = np.trapz(Fv/(v-poles[0])/(v-poles[1]), v)
        exact_solution[k] = 1j* (singlePoleExact(poles[1]) - singlePoleExact(poles[0]))/2/np.imag(poles[0])
    
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(7,6))
    gs = gridspec.GridSpec(2,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    gs.update(left=0.2, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    fig.patch.set_facecolor('white')
    ax = []
    for i in range(0,2):
        ax.append(plt.subplot(gs[i])) 
    
    ax[0].plot(gamma, np.real(approx_calc),linewidth=2)
    ax[0].plot(gamma, np.real(naive_calc),'.')
    ax[0].plot(gamma, np.real(exact_solution),'--',linewidth=2,color='C3')
    ax[1].plot(gamma, np.imag(approx_calc),linewidth=2)
    ax[1].plot(gamma, np.imag(naive_calc),'.')
    ax[1].plot(gamma, np.imag(exact_solution),'--',linewidth=2,color='C3')
    ax[1].plot(gamma, np.imag(plemelj_calc),'k--',linewidth=2)
    ax[1].set_xlabel('$\gamma$')
    ax[0].set_ylabel('Real')
    ax[1].set_ylabel('Imag')
    ax[1].legend(['Pole Refined', 'Naive', 'Exact', 'Plemelj'],loc='lower left',bbox_to_anchor=(.4,1.23,3,4))
    
    for k in range(2):
        ax[k].set_xscale('log')
        ax[k].set_xlim(np.min(gamma), np.max(gamma))
        ax[k].set_xticks([1e-5,1e-3,1e-1,1e1])
        ax[k].grid()
    ax[0].set_xticklabels([])
    
    ax[0].set_ylim(-200,6000)
    ax[1].set_ylim(-.8e-11,.8e-11)
    ax[1].set_yticks([-.5e-11,0,.5e-11])
    ax[1].set_yticklabels(['-5e-12', '0', '5e-12'])
    
    
    # fig.savefig('Documentation/figures/Gaussian_vs_Plemelj_changeGamma_doublePole_1e-5.png',format='png')
    return

# Simple "hand" calculation test showing that this works as intended using Mathematica
def test_calcUs_Maxwellian_handCalc():
    kpar = 0.5
    kperp = 1.5
    omega = 10.
    vth = 1.6e6
    nmax = 2
    rho_avg = 0.054
    Oc = 1.7e5
    nu = 1.2
    
    Us_test = calcUs_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu)
    Us_exact = -2.64961e-6-1j*1.9848e-9   # From Mathematica
    
    print(Us_test)
    print(Us_exact)
    
    return

def test_calcUs_Maxwellian():    
    # Set background parameters based on a reasonable F region plasma 
    Te = 1000/2
    B = 1e-5
    vthe = (2*kB*Te/me)**.5
    Oce = e*B/me
    rho_avge = vthe/Oce/2**.5
    nue = 1.0*100
    nu_ISR = 450e6
    k = 2*math.pi*nu_ISR/c*.01
    
    # Assume 30 deg from parallel (figure out propery terminology for this)
    # Set wavevector based on AMISR parameters
    theta = np.deg2rad(30)
    kpar = k*np.cos(theta)
    kperp = k*np.sin(theta)
    
    # Let omega be from 
    omega = np.linspace(-1000000, 1000000, 101)
    
    # Set the maximum n to use for summation for Bessel function
    nmax = 20
    
    # Make this plot nicer!!!!
    for n in range(0,nmax+1):
        # print(n)
        Us_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, n, rho_avge, Oce, nue)
    
        plt.plot(omega, np.real(Us_exact))
        plt.plot(omega, np.imag(Us_exact))
    
    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('$U_s$')
    return

# Simple "hand" calculation test showing that this works as intended using Mathematica
def test_calcUs_handCalc():
    kpar = 0.5
    kperp = 1.5
    omega = np.array([10.])
    vth = 1.6e6
    nmax = 2
    rho_avg = 0.054
    Oc = 1.7e5
    nu = 1.2
    
    
    
    
    Us_test = calcUs(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu)
    Us_exact = -2.64961e-6-1j*1.9848e-9   # From Mathematica
    
    print(Us_test)
    print(Us_exact)
    

# This function tests the numerical calculation of Us
# Works! Make the plots nicer and do another test for an ion distribution
def test_calcUs():
    # Set background parameters based on a reasonable F region plasma 
    Te = 500
    B = 1e-5
    mi = 16*u
    vthe = (2*kB*Te/me)**.5
    Oce = e*B/me
    rho_avge = vthe/Oce/2**.5
    nue = 100.0
    nu_ISR = 450e6
    k = 2*math.pi*nu_ISR/c    *.01
    nmax = 2
    mesh_n = 1000
    
    # Assume 30 deg from parallel (figure out propery terminology for this)
    # Set wavevector based on AMISR parameters
    theta = np.deg2rad(30)
    kpar = k*np.cos(theta)
    kperp = k*np.sin(theta)
    
    # Let omega be from 
    omega = np.linspace(-100000, 100000, 101)
    # Build a velocity mesh (eventually do a test on what happens when we change dv)
    numVpar = int(1e3)
    numVperp = int(1e3)
    vpar = np.linspace(-6*vthe, 6*vthe, numVpar)
    vperp = np.linspace(0, 6*vthe, numVperp)
    [VVperp, VVpar] = np.meshgrid(vperp, vpar)
    
    f0e = np.exp( -(VVperp**2 + VVpar**2)/vthe**2)/vthe**3/math.pi**1.5
    
    # Calculate approximate solution
    Us_approx = calcUs(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0)
    
    # Calculate exact solution
    Us_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, 20, rho_avge, Oce, nue)
    
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(7,6))
    gs = gridspec.GridSpec(1,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    gs.update(left=0.25, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    fig.patch.set_facecolor('white')
    ax = []
    for i in range(0,1):    
        ax.append(plt.subplot(gs[i])) 
    
    ax[0].plot(omega, np.real(Us_exact),'-',color='C0',linewidth=2,label='Real Exact')
    ax[0].plot(omega, np.imag(Us_exact),'-',color='C3',linewidth=2,label='Imag Exact')
    ax[0].plot(omega, np.real(Us_approx),'o',color='C0',label='Real Approx')
    ax[0].plot(omega, np.imag(Us_approx),'o',color='C3',label='Imag Approx')
    
    ax[0].set_xlabel('$\omega$')
    ax[0].set_ylabel('$U_s$')
    ax[0].grid()
    ax[0].legend(loc='lower right')
    
    fig.savefig('Documentation/figures/test_Us_Maxwellian.png',format='png')
    
    print("vth",vthe)
    print("kperp",kperp)
    print("kpar",kpar)
    print("Oc",Oce)
    print("nu",nue)
    
    return

# Simple "hand" calculation test showing that this works as intended using Mathematica
def test_calcMs_Maxwellian_handCalc():
    kpar = 0.5
    kperp = 1.5
    omega = np.array([10.])
    vth = 1.6e6
    nmax = 2
    rho_avg = 0.054
    Oc = 1.7e5
    nu = 1.2
    
    Us_test = calcUs_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu)
    Ms_test = calcMs_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu, Us_test)
    Us_exact = -2.64961e-6-1j*1.9848e-9   # From Mathematica
    Ms_exact =  2.20802e-6 - 4.32979e-15*1j
    
    
    print(Ms_test)
    print(Ms_exact)
    
def test_calcMs_Maxwellian():    
    # Set background parameters based on a reasonable F region plasma 
    Te = 1000/2
    B = 1e-5
    vthe = (2*kB*Te/me)**.5
    Oce = e*B/me
    rho_avge = vthe/Oce/2**.5
    nue = 1.0*100
    nu_ISR = 450e6
    k = 2*math.pi*nu_ISR/c*.01
    
    # Assume 30 deg from parallel (figure out propery terminology for this)
    # Set wavevector based on AMISR parameters
    theta = np.deg2rad(30)
    kpar = k*np.cos(theta)
    kperp = k*np.sin(theta)
    
    # Let omega be from 
    omega = np.linspace(-50000, 50000, 101)
    
    # Set the maximum n to use for summation for Bessel function
    nmax = 20
    
    # Make this plot nicer!!!!
    for n in range(0,nmax+1):
        # print(n)
        Us_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, n, rho_avge, Oce, nue)
        Ms_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, n, rho_avge, Oce, nue, Us_exact)
    
        plt.plot(omega, np.real(Ms_exact))
        plt.plot(omega, np.imag(Ms_exact))
    print(Ms_exact)
    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('$M_s$')
    return
    
# This function tests the numerical calculation of Ms
# Works! Make the plots nicer and do another test for an ion distribution
def test_calcMs():
    # Set background parameters based on a reasonable F region plasma 
    Te = 500
    B = 1e-5
    vthe = (2*kB*Te/me)**.5
    Oce = e*B/me
    rho_avge = vthe/Oce/2**.5
    nue = 100.0
    nu_ISR = 450e6
    k = 2*math.pi*nu_ISR/c    *.01
    nmax = 2
    mesh_n = 1000
    
    # Assume 30 deg from parallel (figure out propery terminology for this)
    # Set wavevector based on AMISR parameters
    theta = np.deg2rad(30)
    kpar = k*np.cos(theta)
    kperp = k*np.sin(theta)
    
    # Let omega be from 
    omega = np.linspace(-50000, 50000, 101)
    
    # Build a velocity mesh (eventually do a test on what happens when we change dv)
    numVpar = int(1e3)
    numVperp = int(1e3)
    vpar = np.linspace(-6*vthe, 6*vthe, numVpar)
    vperp = np.linspace(0, 6*vthe, numVperp)
    [VVperp, VVpar] = np.meshgrid(vperp, vpar)
    
    f0e = np.exp( -(VVperp**2 + VVpar**2)/vthe**2)/vthe**3/math.pi**1.5
    
    # Calculate approximate solution
    Us_approx = calcUs(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0)
    Ms_approx = calcMs(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0, Us_approx)
    
    # Calculate exact solution
    Us_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, 20, rho_avge, Oce, nue)
    Ms_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, 20, rho_avge, Oce, nue, Us_exact)
    
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(7,6))
    gs = gridspec.GridSpec(1,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    gs.update(left=0.3, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    fig.patch.set_facecolor('white')
    ax = []
    for i in range(0,1):    
        ax.append(plt.subplot(gs[i])) 
    
    ax[0].plot(omega, np.real(Ms_exact),'-',color='C0',linewidth=2,label='Real Exact')
    # ax[0].plot(omega, np.imag(Ms_exact),'-',color='C3',linewidth=2)
    ax[0].plot(omega, np.real(Ms_approx),'o',color='C0',label='Real Approx')
    # ax[0].plot(omega, np.imag(Ms_approx),'o',color='C3')
    
    ax[0].set_xlabel('$\omega$')
    ax[0].set_ylabel('$M_s$')
    ax[0].grid()
    ax[0].legend(loc='lower right')
    
    fig.savefig('Documentation/figures/test_Ms_Maxwellian.png',format='png')
    
    return

# Simple "hand" calculation test showing that this works as intended using Mathematica
# I think this is good enough. But, it's ever so slightly different. Why? Look into this further.
def test_calcChis_Maxwellian_handCalc():
    kpar = 0.5
    kperp = 1.5
    omega = np.array([10.])
    vth = 1.6e6
    nmax = 2
    rho_avg = 0.054
    Oc = 1.7e5
    nu = 1.2
    alpha = 2200
    Te = 1000
    Ts = 400
    
    Us_test = calcUs_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu)
    Chis_test = calcChis_Maxwellian(omega, nu, Us_test, alpha, Te, Ts)
    Us_exact = -2.64961e-6-1j*1.9848e-9   # From Mathematica
    Chis_exact =  1.21e7 - 268.929*1j
    
    
    print(Chis_test)
    print(Chis_exact)

def test_calcChis_Maxwellian():    
    # Set background parameters based on a reasonable F region plasma 
    Te = 1000/2
    B = 1e-5
    vthe = (2*kB*Te/me)**.5
    Oce = e*B/me
    rho_avge = vthe/Oce/2**.5
    nue = 1.0*100
    nu_ISR = 450e6
    k = 2*math.pi*nu_ISR/c*.01
    ne = 1e11
    
    lambdaD = (eps0*kB*Te/(ne*e**2))**.5
    alpha = 1/k/lambdaD
    
    # Assume 30 deg from parallel (figure out propery terminology for this)
    # Set wavevector based on AMISR parameters
    theta = np.deg2rad(30)
    kpar = k*np.cos(theta)
    kperp = k*np.sin(theta)
    
    # Let omega be from 
    omega = np.linspace(-50000, 50000, 101)
    
    # Set the maximum n to use for summation for Bessel function
    nmax = 20
    
    # Make this plot nicer!!!!
    for n in range(0,nmax+1):
        # print(n)
        Us_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, n, rho_avge, Oce, nue)
        Chis_exact = calcChis_Maxwellian(omega, nue, Us_exact, alpha, Te, Te)
    
        plt.plot(omega, np.real(Chis_exact))
        plt.plot(omega, np.imag(Chis_exact))
    plt.xlabel('$\omega$ (rad/s)')
    plt.ylabel('$\chi_s$')
    
    wpe = (ne*e**2/me/eps0)**.5
    print("wpe",wpe)
    
    return

# This function tests the numerical calculation of Chis
def test_calcChis():
    # Set background parameters based on a reasonable F region plasma 
    Te = 500
    B = 1e-5
    vthe = (2*kB*Te/me)**.5
    Oce = e*B/me
    rho_avge = vthe/Oce/2**.5
    nue = 100.0
    nu_ISR = 450e6
    ne = 1e11
    k = 2*math.pi*nu_ISR/c    *.01
    nmax = 0
    mesh_n = 1000
    
    lambdaD = (eps0*kB*Te/(ne*e**2))**.5
    alpha = 1/k/lambdaD
    
    wpe = (ne*e**2/me/eps0)**.5
    
    # Assume 30 deg from parallel (figure out propery terminology for this)
    # Set wavevector based on AMISR parameters
    theta = np.deg2rad(30)
    kpar = k*np.cos(theta)
    kperp = k*np.sin(theta)
    
    # Let omega be from 
    omega = np.linspace(-50000, 50000, 51)
    # omega = np.array([-50000,0,50000])
    
    # Build a velocity mesh (eventually do a test on what happens when we change dv)
    numVpar = int(1e3)
    numVperp = int(1e3)
    vpar = np.linspace(-6*vthe, 6*vthe, numVpar)
    vperp = np.linspace(0, 6*vthe, numVperp)
    [VVperp, VVpar] = np.meshgrid(vperp, vpar)
    
    f0e = np.exp( -(VVperp**2 + VVpar**2)/vthe**2)/vthe**3/math.pi**1.5
    
    # Calculate approximate solution
    Us_approx = calcUs(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0)
    Ms_approx = calcMs(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0, Us_approx)
    Chis_approx = calcChis(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0, Us_approx, wpe)
    
    
    
    # Calculate exact solution
    Us_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, 20, rho_avge, Oce, nue)
    Ms_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, 20, rho_avge, Oce, nue, Us_exact)
    Chis_exact = calcChis_Maxwellian(omega, nue, Us_exact, alpha, Te, Te)
    
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(7,6))
    gs = gridspec.GridSpec(1,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    gs.update(left=0.3, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    fig.patch.set_facecolor('white')
    ax = []
    for i in range(0,1):    
        ax.append(plt.subplot(gs[i])) 
    ax[0].plot(omega, np.real(Chis_exact),'-',color='C0',linewidth=2,label='Real Exact')
    ax[0].plot(omega, np.imag(Chis_exact),'-',color='C3',linewidth=2)
    ax[0].plot(omega, np.real(Chis_approx),'o',color='C0',label='Real Approx')
    ax[0].plot(omega, np.imag(Chis_approx),'o',color='C3')
    
    ax[0].set_xlabel('$\omega$')
    ax[0].set_ylabel('$\chi_s$')
    ax[0].grid()
    # ax[0].legend(loc='lower right')
    
    # fig.savefig('Documentation/figures/test_Ms_Maxwellian.png',format='png')
    
    return

# This function tests the numerical calculation of M and Chi in a more efficient way
def test_calcM_Chi():
    # Set background parameters based on a reasonable F region plasma 
    Te = 500
    B = 1e-5
    vthe = (2*kB*Te/me)**.5
    Oce = e*B/me
    rho_avge = vthe/Oce/2**.5
    nue = 100.0
    nu_ISR = 450e6
    ne = 1e11
    k = 2*math.pi*nu_ISR/c    *.01
    nmax = 2
    mesh_n = 1000
    
    lambdaD = (eps0*kB*Te/(ne*e**2))**.5
    alpha = 1/k/lambdaD
    
    wpe = (ne*e**2/me/eps0)**.5
    
    # Assume 30 deg from parallel (figure out propery terminology for this)
    # Set wavevector based on AMISR parameters
    theta = np.deg2rad(30)
    kpar = k*np.cos(theta)
    kperp = k*np.sin(theta)
    
    # Let omega be from 
    omega = np.linspace(-50000, 50000, 51)
    M_approx = np.zeros_like(omega) + 1j*0.0
    chi_approx = np.zeros_like(omega) + 1j*0.0
    U_approx = np.zeros_like(omega) + 1j*0.0
    # omega = np.array([-50000,0,50000])
    
    # Build a velocity mesh (eventually do a test on what happens when we change dv)
    numVpar = int(1e3)
    numVperp = int(1e3)
    vpar = np.linspace(-6*vthe, 6*vthe, numVpar)
    vperp = np.linspace(0, 6*vthe, numVperp)
    [VVperp, VVpar] = np.meshgrid(vperp, vpar)
    
    f0e = np.exp( -(VVperp**2 + VVpar**2)/vthe**2)/vthe**3/math.pi**1.5
    
    # Calculate approximate solution
    # Us_approx = calcUs(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0)
    # Ms_approx = calcMs(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0, Us_approx)
    # Chis_approx = calcChis(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0, Us_approx, wpe)
    for k in range(len(omega)):
        print("omega:", omega[k])
    # [M_approx,Chi_approx] = calcM_Chi(vpar, vperp, f0e, omega, kpar, kperp, nmax, Oce, nue, mesh_n, 0, wpe)
        
        [M_approx[k], chi_approx[k], U_approx[k]] = calcU_M_chi(vpar, vperp, f0e, omega[k], kpar, kperp, 0, nmax, Oce, nue, mesh_n, 0, wpe, 0.0+1j*0.0, 0.0+1j*0.0, 0.0+1j*0.0)[0:3]
    
    # Calculate exact solution
    Us_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, 20, rho_avge, Oce, nue)
    Ms_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, 20, rho_avge, Oce, nue, Us_exact)
    chis_exact = calcChis_Maxwellian(omega, nue, Us_exact, alpha, Te, Te)
    
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(7,6))
    gs = gridspec.GridSpec(1,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    gs.update(left=0.3, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    fig.patch.set_facecolor('white')
    ax = []
    for i in range(0,1):    
        ax.append(plt.subplot(gs[i])) 
    ax[0].plot(omega, np.real(Ms_exact),'-',color='C0',linewidth=2,label='Real Exact')
    ax[0].plot(omega, np.imag(Ms_exact),'-',color='C3',linewidth=2)
    ax[0].plot(omega, np.real(M_approx),'o',color='C0',label='Real Approx')
    ax[0].plot(omega, np.imag(M_approx),'o',color='C3')
    
    ax[0].set_xlabel('$\omega$')
    ax[0].set_ylabel('$M_s$')
    ax[0].grid()
    ax[0].legend(loc='lower right')
    
    
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig2 = plt.figure(2, figsize=(7,6))
    gs2 = gridspec.GridSpec(1,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    gs2.update(left=0.3, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    fig2.patch.set_facecolor('white')
    ax2 = []
    for i in range(0,1):    
        ax2.append(plt.subplot(gs2[i])) 
    ax2[0].plot(omega, np.real(chis_exact),'-',color='C0',linewidth=2,label='Real Exact')
    ax2[0].plot(omega, np.imag(chis_exact),'-',color='C3',linewidth=2)
    ax2[0].plot(omega, np.real(chi_approx),'o',color='C0',label='Real Approx')
    ax2[0].plot(omega, np.imag(chi_approx),'o',color='C3')
    
    ax2[0].set_xlabel('$\omega$')
    ax2[0].set_ylabel('$\chi_s$')
    ax2[0].grid()
    ax2[0].legend(loc='lower right')
    
    # fig.savefig('Documentation/figures/test_Ms_Maxwellian.png',format='png')
    
    return

# M and Chi function works. Now to test using an ion paramter regime since that's what we will be doing these calculations for. 
# Maybe also do some sort of test looking at how all this changes with nmax
def test_calcM_Chi_ions():
    # Set background parameters based on a reasonable F region plasma 
    Ti = 1000
    B = 1e-5
    mi = 16*u
    vthi = (2*kB*Ti/mi)**.5
    Oci = e*B/mi
    rho_avgi = vthi/Oci/2**.5
    nui = 1.0
    nu_ISR = 450e6
    ni = 1e11
    k = 2*math.pi*nu_ISR/c
    nmax = 0
    mesh_n = 1000
    
    Te = 2000.
    ne = 1e11
    lambdaD = (eps0*kB*Te/(ne*e**2))**.5
    alpha = 1/k/lambdaD
    
    wpi = (ni*e**2/mi/eps0)**.5
    
    # Assume 30 deg from parallel (figure out propery terminology for this)
    # Set wavevector based on AMISR parameters
    theta = np.deg2rad(30)
    kpar = k*np.cos(theta)
    kperp = k*np.sin(theta)
    
    # Let omega be from 
    omega = np.linspace(-10000, 10000, 401)
    
    # Build a velocity mesh (eventually do a test on what happens when we change dv)
    numVpar = int(1e3)
    numVperp = int(1e3)
    
    gamma = nui/kpar/vthi
    dv = 1e-2*vthi
    vpar = np.arange(-4*vthi,4*vthi+dv,dv)
    vperp = np.arange(0,4*vthi+dv,dv)
    print("npar",len(vpar))
    print("nperp",len(vperp))
    
    print("gamma/vthi",nui/kpar/vthi)
    print("dv/vth",(vpar[1]-vpar[0])/vthi)
    [VVperp, VVpar] = np.meshgrid(vperp, vpar)
    
    f0i = np.exp( -(VVperp**2 + VVpar**2)/vthi**2)/vthi**3/math.pi**1.5
    
    
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(14,6))
    gs = gridspec.GridSpec(2,2)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    gs.update(left=0.3, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    fig.patch.set_facecolor('white')
    ax = []
    for i in range(0,4):    
        ax.append(plt.subplot(gs[i])) 
        
    start_time = time.time()
    for n in range(nmax,nmax+1):
        # Calculate approximate solution
        
        [M_i_approx,Chi_i_approx] = calcM_Chi(vpar, vperp, f0i, omega, kpar, kperp, n, Oci, nui, mesh_n, 0, wpi)
        ax[0].plot(omega, np.real(M_i_approx),'-',linewidth=2)
        ax[1].plot(omega, np.imag(M_i_approx),'-',linewidth=2)
        
        ax[2].plot(omega, np.real(Chi_i_approx),'-',linewidth=2)
        ax[3].plot(omega, np.imag(Chi_i_approx),'-',linewidth=2)
   
   
    end_time = time.time() - start_time
    print(end_time/len(omega)/(nmax+1))
    # print("Run Time:", end_time)
    ax[2].set_xlabel('$\omega$')
    ax[3].set_xlabel('$\omega$')
    ax[0].set_ylabel('$M_s$')
    ax[2].set_ylabel('$\chi_s$')
    ax[0].set_title('Real')
    ax[1].set_title('Imag')
    
    for k in range(0,4):
        ax[k].grid()
    # fig.savefig('Documentation/figures/test_Ms_Maxwellian.png',format='png')
    

    return

def test_calcSpectra_handCalc():
    chi_e = 12. + 1j*2.0
    chi_i = -3 + 1j*4.0
    M_i = 3.
    M_e = 5.
    
    S_approx = calcSpectra(M_i, M_e, chi_i, chi_e)
    S_exact = 8  # From a hand calculation
    print(S_approx)
    

# Run the testing functions
# test_getVInterp()
# test_getIntegrand()
# test_plemelj()
# test_poleIntegrate_ones_vs_Plemelj_changeGamma()
# test_poleIntegrate_Gaussian_vs_Plemelj_changeGamma()
# test_poleIntegrate_Gaussian_vs_Plemelj_changeGamma_doublePole()
# test_calcUs_Maxwellian_handCalc()
# test_calcUs_Maxwellian()
# test_calcUs_handCalc()
# test_calcUs()
# test_calcMs_Maxwellian_handCalc()
# test_calcMs_Maxwellian()
# test_calcMs()
# test_calcChis_Maxwellian_handCalc()
# test_calcChis_Maxwellian()
# test_calcChis()
# test_calcM_Chi()
# test_calcM_Chi_ions()
test_calcSpectra_handCalc()