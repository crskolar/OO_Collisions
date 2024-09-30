# The set of functions needed to calculate the susceptibility from an arbitrary distribution function

import numpy as np
import math
import scipy.special as sp
import scipy.constants as const

###############################################################################

################### Functions for evaluating pole integrals ###################

###############################################################################

# Get linear interpolation coefficients for 1D or 2D arrays 
# Note that this only works in the one 2D array direction (iterating through all the rows)
# So, you need to make sure that you are properly indexing in the parallel direction
def getLinearInterpCoeff(x, y):
    # Set everything up based on if we have 1d or 2d data
    if len(np.shape(y)) == 2:  # 2D data
        [numPar, numPerp] = np.shape(y)
        numElements = numPar - 1
        a = np.zeros((numElements, numPerp))
        b = np.zeros((numElements, numPerp))
    elif len(np.shape(y)) == 1:  # 1D data (I honestly don't remember why I implemented this, I only use it with 2D)
        numElements = len(y) - 1
        a = np.zeros(numElements)
        b = np.zeros(numElements)

    # Iterate through numElements
    for i in range(numElements):
        a[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
        b[i] = y[i]-a[i]*x[i]
    return a,b

# Function for the exact solution to the indefinite integral of (a*x+b)/(x-z)
def p1IndefiniteIntegral(a, b, z, x):
    return a*(x-z)+(b+a*z)*np.log(x-z)

# Function for the exact solution to the indefinite integral of (a*x+b)/(x-z)^2
def p2IndefiniteIntegral(a, b, z, x):
    return (-b-a*z)/(x-z)+a*np.log(x-z)

# Function for the exact solution to the indefinite integral of (a*x+b)/(x-z)/(x-z*)
def pstarIndefiniteIntegral(a, b, z, x):
    return ((b+a*np.conjugate(z))*np.log(x-np.conjugate(z)) - (b+a*z)*np.log(x-z))*1j/np.imag(z)/2.

# Conduct parallel pole integrations based on choice of indefinite integral function above
# We are evaluating the indefinite integral over each line segment
# Then, we calculate the whole integral by taking the summation of each line segement value
def interpolatedIntegral(a, b, z, x, integralFunction):
    integratedValue = 0.0 + 1j*0.0  # Initialize a complex valued zero
    # Iterate through each element
    for i in range(len(a)):
        # Evaluate the indefinite integral in each line segment by evaluating at the edge points
        integratedValue += integralFunction(a[i], b[i], z, x[i+1]) - integralFunction(a[i], b[i], z, x[i])
    return integratedValue

###############################################################################

################### Functions for calculating interim values ##################

###############################################################################

# This function calculates the perpendicular and parallel wavenumbers
# Inputs are the radar frequency in Hz and the aspect angle in degrees
# This code assumes that 0 is parallel to the magnetic field and 90 is perpendicular
# Outputs wavenumber components in rad/m
def getK(nu_radar, angle):
    k_ISR = 2*2*math.pi*nu_radar/const.c
    theta = np.deg2rad(angle)
    kpar = k_ISR*np.cos(theta)
    kperp = k_ISR*np.sin(theta)
    return kperp, kpar

# This function calculates the gyrofrequency (Oc)
# Inputs are the charge in C, the magnetic field strength in T, and mass in kg
# Outputs gyrofrequency in rad/s
def getGyroFrequency(q, B, m):
    return np.abs(q*B/m)  # Assumes absolute value in case you gave a signed charge 

# This function calculates the thermal velocity of a species
# Intputs are temperature in K and mass in kg
# Output is thermal velocity in m/s
def getVth(T, m):
    return (2*const.k*T/m)**.5

# This function calculates the plasma frequency of a species
# Inputs are density (n) in m^-3, charge in C, and mass in kg
# Output is plasma frequency in rad/s
def getPlasmaFrequency(n, q, m):
    return (n*q**2/m/const.epsilon_0)**.5

# This function calculates the average Larmor radius
# Inputs are thermal velocity in m/s and gyrofrequency in rad/s
# Output is average Larmor radius in m
def getLarmorRadius_avg(vth, Oc):
    return vth/Oc/2**.5    

# This function calculates the Debye length based on the electron density and temperature
# Inputs are electron number density in m^-3 and electron temperature in K
# Output is Debye length in m
def getDebyeLength(ne, Te):
    return (const.epsilon_0*const.k*Te/ne/const.e**2)**.5

# Calculate alpha, which is the incoherent scattering parameter
def getAlpha(kperp, kpar, ne, Te):
    lambdaD = getDebyeLength(ne, Te)
    k_ISR = (kperp**2+kpar**2)**.5
    return 1/k_ISR/lambdaD

# This function makes the velocity mesh (in m/s)
# Input the order of the delta v in each direction
# So 0 means 1 point per thermal velocity, -1 is 10 points, -2 is 100, etc.
# extent is the number of thermal velocities you want your velocity space to extend to
# In general, 4 is a good value for the extents (esp. for Maxwellians)
# -2 is a good value for dvperp_order
# Start at -2.3 for dvpar_order and decrease as needed to ensure proper convergence
# The outputs are the 2D velocity meshes (VVperp and VVpar) and the 1D velocity meshes (vperp and vpar)
def makeVelocityMesh(vth, dvperp_order, dvpar_order, extentPerp, extentPar):
    dvpar = 10**dvpar_order*vth
    vpar = np.arange(-extentPar*vth,extentPar*vth+dvpar,dvpar)
    dvperp = 10**dvperp_order*vth
    vperp = np.arange(0,extentPerp*vth+dvperp,dvperp)

    [VVperp, VVpar] = np.meshgrid(vperp, vpar)
    return VVperp, VVpar, vperp, vpar

###############################################################################

############# Calculate numerical approximations for U and chi ################

###############################################################################

# This function calculates the individual perpendicular integral summation term for U and M for a specific n
# We combine U and M because they do the same calculation but just have different inputs
# For U, use p1 (first order pole at z)
# For M, use pstar (two first poles at z and z*)
def calcPerpSum_U_M(n, kperp, vperp, Oc, p):
    return np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*p,vperp)

# This function calculates the individual perpendicular integral summation term for chi for a specific n
def calcPerpSum_chi(n, kperp, kpar, vperp, Oc, p1, p2):
    return np.trapz(vperp*p2*sp.jv(n,kperp*vperp/Oc)**2*(-1),vperp) + n*kperp/kpar*np.trapz(p1*sp.jv(n,kperp*vperp/Oc)*(sp.jv(n-1,kperp*vperp/Oc)-sp.jv(n+1,kperp*vperp/Oc)),vperp)

# This function calculates U from the summation terms
def calcU_fromSum(sum_U, kpar, nu):
    if len(np.shape(sum_U)) == 2:
        return -2*math.pi*1j*nu/kpar*np.sum(sum_U,axis=0)
    elif len(np.shape(sum_U)) == 1:
        return -2*math.pi*1j*nu/kpar*sum_U

def calChi_fromSum(sum_chi, kperp, kpar, wp, U):
    if len(np.shape(sum_chi)) == 2:
        return np.sum(sum_chi,axis=0)*2*math.pi*wp**2/(kpar**2+kperp**2)/(1+U)
    elif len(np.shape(sum_chi)) == 1:
        return sum_chi*2*math.pi*wp**2/(kpar**2+kperp**2)/(1+U)

# This is a serial function that just calculates the susceptibility
# Inputs are:
# vperp: 1D array of perpendicular velocity in m/s
# vpar: 1D array of parallel velocity in m/s
# f0: 2D normalized distribution function in m^-3 s^3. Note that for getLinearInterpCoeff to work, f0 must be set such that the f0[i] or f0[i,:] indexes the parallel direction
# omega: 1D array of angular frequencies in rad/s you want the calculation run at
# nu_radar: Radar frequency in Hz
# theta: Aspect angle in degrees
# q: Particle charge in C
# B: Magnetic field strength in T
# m: Particle mass in kg
# nu: Collision frequency in Hz
# nDens: Number density in m^-3
# nmax: Maximum summation term to consider
def calcChi(vperp, vpar, f0, omega, nu_radar, theta, q, B, m, nu, nDens, nmax):
    # Get wavenumber from radar frequency and angle
    [kperp, kpar] = getK(nu_radar, theta)

    # Get the gyrofrequency
    Oc = getGyroFrequency(q, B, m)

    # Get plasma frequency
    wp = getPlasmaFrequency(nDens, q, m)

    # Get linear interpolation coefficients
    [a, b] = getLinearInterpCoeff(vpar, f0)

    # Initialize a bunch of complex valued zeros for summation terms
    sum_U = np.zeros_like(omega) + 1j*0.0
    sum_chi = np.zeros_like(omega) + 1j*0.0

    # Iterate from -nmax to nmax
    for n in range(-nmax, nmax+1):
        print('n=%d' % (n))
        # Calculate the pole
        z = (omega - n*Oc - 1j*nu)/kpar

        # Iterate through omega to get summation terms for each omega
        for k in range(0, len(omega)):
            # Do the parallel integrals for the first and second order poles
            # 1: A single pole at z with order 1
            # 2: A single pole at z with order 2
            p1 = interpolatedIntegral(a, b, z[k], vpar, p1IndefiniteIntegral)
            p2 = interpolatedIntegral(a, b, z[k], vpar, p2IndefiniteIntegral)
        
            # Use trapz to do the perpendicular integrals and perform the summations for U and chi
            sum_U[k] += calcPerpSum_U_M(n, kperp, vperp, Oc, p1)
            sum_chi[k] += calcPerpSum_chi(n, kperp, kpar, vperp, Oc, p1, p2)
    U = calcU_fromSum(sum_U, kpar, nu)
    chi = calChi_fromSum(sum_chi, kperp, kpar, wp, U)
        
    return chi

###############################################################################

################ Exact solutions for a Maxwellian distribution ################

###############################################################################

# This function calculates the exact collisional term, U, for a Maxwellian distribution
# omega is a 1D array
# Everything else is a scalar
def calcU_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu):
    # Calculate yn
    U = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from -nmax to nmax
    for n in range(-nmax,nmax+1):
        # Calculate yn
        yn = (omega - n*Oc-1j*nu)/kpar/vth
        U += sp.ive(n, kperp**2*rho_avg**2)*(2*sp.dawsn(yn) + 1j*math.pi**.5*np.exp(-yn*yn))
    # Multiply by everything else and return
    return U*1j*nu/kpar/vth

# This function calculates the exact susceptibility, chi, for a Maxwellian distribution
# omega and U are 1D arrays
# Everything else is a scalar
def calcChi_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu, alpha, U, Te, Ts):
    chi = np.zeros_like(omega) + 1j*0.0
    for n in range(-nmax,nmax+1):
        yn = (omega - n*Oc-1j*nu)/kpar/vth
        chi += sp.ive(n, kperp**2*rho_avg**2)*(2*sp.dawsn(yn) + 1j*math.pi**.5*np.exp(-yn**2)  )
    return (1-(omega-1j*nu)*chi/kpar/vth)*alpha**2*Te/Ts/(1+U)

###############################################################################

########################### Distribution Functions ############################

###############################################################################

# Calculate a normalized Maxwellian distribution in cylindrical coordinates
def maxwellian_norm(vperp, vpar, vth):
    return np.exp(-(vperp**2+vpar**2)/vth**2)/vth**3/math.pi**1.5
