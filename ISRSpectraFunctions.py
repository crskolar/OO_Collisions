import numpy as np
import math
import scipy.special as sp
import multiprocessing as mp
from functools import partial
import logging
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
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

############## Calculate numerical approximations for U, M, chi ###############

###############################################################################

# Initialize the logger
def initialize_logger(fileName, fileDir, num_processors):
    logging.root.handlers = []
    logging.basicConfig(level=logging.DEBUG, 
                        format='[%(asctime)s] %(message)s', 
                        datefmt='%m/%d/%Y %H:%M:%S',
                        handlers=[logging.FileHandler(fileDir+fileName+'.log',mode='w'),
                                  logging.StreamHandler()])
    logging.info('Initializing log for %s using %d processors' % (fileName, num_processors))

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
    
def calcM_fromSum(sum_M, kpar, nu, U):
    if len(np.shape(sum_M)) == 2:
        return (np.sum(sum_M,axis=0)*2*math.pi/kpar**2 -np.abs(U)**2/nu**2)*nu/np.abs(1+U)**2
    elif len(np.shape(sum_M)) == 1:
        return (sum_M*2*math.pi/kpar**2 -np.abs(U)**2/nu**2)*nu/np.abs(1+U)**2
    
def calChi_fromSum(sum_chi, kperp, kpar, wp, U):
    if len(np.shape(sum_chi)) == 2:
        return np.sum(sum_chi,axis=0)*2*math.pi*wp**2/(kpar**2+kperp**2)/(1+U)
    elif len(np.shape(sum_chi)) == 1:
        return sum_chi*2*math.pi*wp**2/(kpar**2+kperp**2)/(1+U)


# Calculate the summation terms for U, M, and chi. 
# Takes an input of one n value
# Returns one set of values for the summation terms corresponding to n and -n for U, chi, and M   
# a and b are arrays of the slope and y-intercept of the linear interpolation elements
# omega is a 1D array
# Everything else is a scalar
def calcSumTerms(nBase, vpar, vperp, a, b, omega, kpar, kperp, Oc, nu):
    start_time = time.time()
    proc = mp.Process()

    # Initialize a bunch of complex valued zeros
    sum_U = np.zeros_like(omega) + 1j*0.0
    sum_M = np.zeros_like(omega) + 1j*0.0
    sum_chi = np.zeros_like(omega) + 1j*0.0
    
    # Iterate through the two signs. Summation term needs to be symmetric
    # So for each individual n, we calculate the n and -n summation value
    for sgn in [-1.0, 1.0]:
        n = sgn*nBase

        # Calculate the pole
        z = (omega - n*Oc - 1j*nu)/kpar

        # Iterate through omega to get summation terms for each omega
        for k in range(0, len(omega)):
            # Do the parallel integrals for the three types of poles that show up in the calculations
            # 1: A single pole at z with order 1
            # 2: A single pole at z with order 2
            # 3: A double pole at z and z* (each first order)
            p1 = interpolatedIntegral(a, b, z[k], vpar, p1IndefiniteIntegral)
            p2 = interpolatedIntegral(a, b, z[k], vpar, p2IndefiniteIntegral)
            pstar = interpolatedIntegral(a, b, z[k], vpar, pstarIndefiniteIntegral)
            
            # Use trapz to do the perpendicular integrals and perform the summations for U, M, and chi
            # dummy = 
            sum_U[k] += calcPerpSum_U_M(n, kperp, vperp, Oc, p1)
            # np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*p1,vperp)
            sum_M[k] += calcPerpSum_U_M(n, kperp, vperp, Oc, pstar)
            # np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*pstar,vperp)
            sum_chi[k] += calcPerpSum_chi(n, kperp, kpar, vperp, Oc, p1, p2)
            # np.trapz(vperp*p2*sp.jv(n,kperp*vperp/Oc)**2*(-1),vperp) + n*kperp/kpar*np.trapz(p1*sp.jv(n,kperp*vperp/Oc)*(sp.jv(n-1,kperp*vperp/Oc)-sp.jv(n+1,kperp*vperp/Oc)),vperp)
        # If n is 0, break the loop so we aren't double counting the zeroth order term
        if n == 0:
            break
    end_time = time.time() - start_time
    logging.info('n = %d is done on %s in %.2es' % (nBase, proc.name, end_time))
    return sum_U, sum_M, sum_chi

# A parallelized function that will call calcSumTerms and iterate from -nmax to nmax
# Will output all of the individual summation terms as an array
def calcSumTerms_par(num_processors, nStart, nmax, vpar, vperp, a, b, omega, kpar, kperp, Oc, nu, m, wp, fileDir, fileName):
    # Save the wavenumbers, collision frequency, mass, and plasma frequency. 
    # These are used to calculate U, M, and chi later
    param_file = open(fileDir + fileName + '_param.txt', 'w')
    param_file.write('kperp: %.18e\nkpar: %.18e\nnu: %.18e\nm: %.18e\nwp: %.18e' % (kperp, kpar, nu, m, wp))
    param_file.close()
    
    # Save omega
    np.savetxt(fileDir + fileName + '_omega.txt', omega, fmt='%.18e')
    
    # Make an array for all n values we want to consider
    nArray = np.arange(nStart, nmax+1)
    
    # Check to make sure that the number of processors is <= to the available number on computer
    if num_processors > mp.cpu_count():
        num_processors = mp.cpu_count()  # If too many called, default to max available
    
    # Make an iterable function that only changes n in calcSumTerms and leaves everything else a constant
    caclSumTerms_iterable = partial(calcSumTerms,vpar=vpar, vperp=vperp, a=a, b=b, omega=omega, kpar=kpar, kperp=kperp, Oc=Oc, nu=nu)
    
    # Set pool based on number of processors
    with mp.Pool(processes=num_processors) as pool:
        # Perform the parallel calculation
        out = np.array(pool.map(caclSumTerms_iterable, nArray))
    
    # Output to previous dumps everything into a single 3D array that is of the following shape:
    # nmax+1 x 3 x len(omega)
    # Return the values in appropriate 2D summation arrays that will be of shape:
    # nmax+1 x len(omega)
    sum_U =out[:,0,:]
    sum_M = out[:,1,:]
    sum_chi = out[:,2,:]
    
    # Save the data so that we have it for the future
    np.savetxt(fileDir + fileName + '_sum_U.txt', sum_U, fmt='%.18e')
    np.savetxt(fileDir + fileName + '_sum_M.txt', sum_M, fmt='%.18e')
    np.savetxt(fileDir + fileName + '_sum_chi.txt', sum_chi, fmt='%.18e')
    
    return sum_U, sum_M, sum_chi

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


# Calculate U, M, and chi from their summation terms
# Input summations are assumed to be of the shape nmax+1 x len(omega) as per the output of the previous function
# Outputs U, M, and chi as 1D arrays (as functions of omega)
def calcFromSums(sum_U, sum_M, sum_chi, kpar, kperp, nu, wp):
    U = calcU_fromSum(sum_U, kpar, nu)
    M = calcM_fromSum(sum_M, kpar, nu, U)
    chi = calChi_fromSum(sum_chi, kperp, kpar, wp, U)
    return U, M, chi

# Define a metric for iterative error
# Used to convergence checking with the summation
def calcIterError(old, new):
    return np.max(np.abs(old - new)) # For now, using LInfinity norm of difference between old and new

# Load summation and parameter data based on all the outputs of calcSumTerms_par 
def loadSumData(fileName):
    # Load summation data
    sum_U = np.loadtxt(fileName+'_sum_U.txt', dtype=np.complex128)
    sum_M = np.loadtxt(fileName+'_sum_M.txt', dtype=np.complex128)
    sum_chi = np.loadtxt(fileName+'_sum_chi.txt', dtype=np.complex128)
    
    # Load omega data
    omega = np.loadtxt(fileName + '_omega.txt')
    
    # Load parameter data
    param_data_file = open(fileName+'_param.txt')
    param_data_lines = param_data_file.readlines()
    kperp = float(param_data_lines[0][7:-1])
    kpar = float(param_data_lines[1][6:-1])
    nu = float(param_data_lines[2][3:-1])
    m = float(param_data_lines[3][3:-1])
    wp = float(param_data_lines[4][4:])
    param_data_file.close()
    
    return sum_U, sum_M, sum_chi, omega, kperp, kpar, nu, m, wp

# Check the spectrum for convergence in terms of number of summation terms (n)
# We will do this by testing differences with each succesive summation in U, M, and chi
# Input is the base file name for all of the data (including any file directory information)
def checkSumConvergence(fileName, TOL):
    [sum_U, sum_M, sum_chi, omega, kperp, kpar, nu, m, wp] = loadSumData(fileName)
    
    # Get nmax
    nmax = len(sum_U)-1
    nArray = np.arange(1,nmax+1)
    
    # Initialize the errors
    Re_U_error = np.zeros(nmax)
    Im_U_error = np.zeros(nmax)
    Re_M_error = np.zeros(nmax)
    Re_chi_error = np.zeros(nmax)
    Im_chi_error = np.zeros(nmax)
    
    # Iterate through n
    for n in range(nmax+1):
        # Calculate U, M, and chi up to each n
        [U, M, chi] = calcFromSums(sum_U[:(n+1)], sum_M[:(n+1)], sum_chi[:(n+1)], kpar, kperp, nu, 1.0) # Might be able to optimize this, but works for now
        
        if n != 0:  # If n isn't 0, then do the convergence calculation between current and previous iteration
            Re_U_error[n-1] = calcIterError(np.real(U_old), np.real(U))
            Im_U_error[n-1] = calcIterError(np.imag(U_old), np.imag(U))
            Re_M_error[n-1] = calcIterError(np.real(M_old), np.real(M))
            Re_chi_error[n-1] = calcIterError(np.real(chi_old), np.real(chi))
            Im_chi_error[n-1] = calcIterError(np.imag(chi_old), np.imag(chi))
            
        # Set the old values to be the current values
        U_old = U*1.0
        M_old = M*1.0
        chi_old = chi*1.0
        
    # Initialize figure
    font = {'size'   : 22}
    mpl.rc('font', **font)
    figConverge = plt.figure(1, figsize=(8,6))
    gs = gridspec.GridSpec(1,1)
    gs.update(left=0.17, right=.99, bottom=.14, top=.95, wspace=0.15, hspace=.015)
    figConverge.patch.set_facecolor('white')
    ax = plt.subplot(gs[0])
    
    # Plot all errors
    ax.plot(nArray, Re_U_error,'--',linewidth=2,color='C0',label='Re($U$)')
    ax.plot(nArray, Im_U_error,'--',linewidth=2,color='C1',label='Im($U$)')
    ax.plot(nArray, Re_M_error,'--',linewidth=2,color='C3',label='Re($M$)')
    ax.plot(nArray, Re_chi_error,'--',linewidth=2,color='C8',label='Re($\\chi$)')
    ax.plot(nArray, Im_chi_error,'--',linewidth=2,color='C9',label='Im($\\chi$)')
        
    # Make figure look nice
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    ax.set_xlabel('$n_{max}$')
    ax.set_ylabel('Iterative Error')
    ax.legend(ncols=1,loc='lower left')#,bbox_to_anchor=  (1.04,-.05,.4,.8))
    
    # Text output on convergence status
    if (np.array([Re_U_error[-1], Im_U_error[-1], Re_M_error[-1], Re_chi_error[-1], Im_chi_error[-1]])<TOL).all():
        print(fileName + " has converged.")
        converged = True
    else:
        print(fileName + " has not converged.")
        converged = False
    
    # Save figure
    figConverge.savefig(fileName+'_convergenceCheck.png',format='png')
    plt.close(figConverge)
    return converged

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

# This function calculates the exact modified distribution, M, for a Maxwellian distribution
# omega and U are 1D arrays
# Everything else is a scalar
def calcM_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu, U):
    M = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from -nmax to nmax
    for n in range(-nmax,nmax+1):
        yn = (omega - n*Oc-1j*nu)/kpar/vth
        M += sp.ive(n, kperp**2*rho_avg**2)*(math.pi**.5*np.real(np.exp(-yn**2))+2*np.imag(sp.dawsn(yn)))    
    # Multiply and add everything else and return
    return M/kpar/vth/np.abs(1+U)**2-np.abs(U)**2/nu/np.abs(1+U)**2

# This function calculates the modified distribution, M, as formulated by Froula
# This is correct in the limit of nu goes to zero, but is incorrect in collisional cases
def calcM_Maxwellian_Froula(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu, U):
    M = np.zeros_like(omega) + 1j*0.0

    # Iterate from -nmax to nmax
    for n in range(-nmax,nmax+1):
        yn = (omega - n*Oc-1j*nu)/kpar/vth
        M += sp.ive(n, kperp**2*rho_avg**2)*np.exp(-yn**2)
    # Multiply and add everything else and return
    return M*math.pi**.5/kpar/vth/np.abs(1+U)**2-np.abs(U)**2/nu/np.abs(1+U)**2

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

############## Exact solutions for a bi-Maxwellian distribution ###############

###############################################################################

# This function calculates the exact collisional term, U, for a bi-Maxwellian distribution
# omega is a 1D array
# Everything else is a scalar
def calcU_biMax(omega, kpar, kperp, vthpar, nmax, rho_avg, Oc, nu):
    # Initialize U
    U = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from -nmax to nmax
    for n in range(-nmax, nmax+1):
        # Calculate yn
        yn = (omega - n*Oc-1j*nu)/kpar/vthpar
        
        # Do summation term in U
        U += sp.ive(n, kperp**2*rho_avg**2)*(2*sp.dawsn(yn) + 1j*math.pi**.5*np.exp(-yn**2))
    return U*1j*nu/kpar/vthpar

# This function calculates the exact modified distribution function, M, for a bi-Maxwellian distribution
# omega and U are 1D arrays
# Everything else is a scalar
def calcM_biMax(omega, kpar, kperp, vthpar, nmax, rho_avg, Oc, nu, U):
    # Initialize M
    M = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from -nmax to nmax
    for n in range(-nmax, nmax+1):
        # Calculate yn and ynstar
        yn = (omega - n*Oc-1j*nu)/kpar/vthpar
        ynstar = (omega - n*Oc+1j*nu)/kpar/vthpar
        
        # Do summation term for M
        M += sp.ive(n, kperp**2*rho_avg**2)*np.exp(-yn**2)
        
        #sp.ive(n,kperp**2*rho_avg**2)*np.exp(-ynstar**2)*(np.exp(4*1j*nu*(omega-n*Oc)/kpar**2/vthpar**2)*sp.erfc(1j*yn)+sp.erf(1j*ynstar)+1)
    return np.real(M*math.pi**.5/kpar/vthpar/np.abs(1+U)**2-np.abs(U)**2/nu/np.abs(1+U)**2)    #  np.real(M*math.pi**0.5/2/kpar/vthpar/np.abs(1+U)**2-np.abs(U)**2/nu/np.abs(1+U)**2)

# omega and U are 1D arrays
# Everything else is a scalar
def calcChi_biMax(omega, kpar, kperp, vthpar, vthperp, nmax, rho_avg, Oc, nu, U, alpha, Te, Tpar):
    # Initialize chi
    chi = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from -nmax to nmax
    for n in range(-nmax, nmax+1):
        # Calculate yn
        yn = (omega - n*Oc-1j*nu)/kpar/vthpar
        
        # Do summation term
        chi += sp.ive(n, kperp**2*rho_avg**2)*(1-(n*Oc*vthpar/kpar/vthperp**2+yn)*(2*sp.dawsn(yn)+1j*math.pi**.5*np.exp(-yn**2)))
        
    return chi*Te*alpha**2/Tpar/(1+U)

# Modified distribution still needs work. But this whole section is unimportant

###############################################################################

########################### Calculate ISR spectrum ############################

###############################################################################

# Calculate the spectra based on modified distributions and susceptabilities of ions and electrons
# Currently assumes only one ion species
def calcSpectra(M_i, M_e, chi_i, chi_e):
    # Calculate the dielectric function
    eps = 1 + chi_e + chi_i
    return np.real(2*np.abs(1-chi_e/eps)**2*M_e+2*np.abs(chi_e/eps)**2*M_i)


###############################################################################

########################### Distribution Functions ############################

###############################################################################

# Calculate a normalized Maxwellian distribution in cylindrical coordinates
def maxwellian_norm(vperp, vpar, vth):
    return np.exp(-(vperp**2+vpar**2)/vth**2)/vth**3/math.pi**1.5

# Calculate a normalized toroidal distribution in cylindrical coordinates
def toroidal_norm(vperp, vpar, vthperp, vthpar, Dstar):
    Cperp = np.abs(vperp)/vthperp
    Cpar = vpar/vthpar
    return sp.ive(0,2*Dstar*Cperp)*np.exp(-Cpar**2-(Cperp-Dstar)**2)/vthperp**2/vthpar/math.pi**1.5

# Calculate a normalized kappa distribution in cylindrical coordinates
# Note that we are using a loggamma approach so that we can use higher kappa values without overflow errors
def kappa_norm(vperp, vpar, vth, kappa):
    return (1+(vperp**2+vpar**2)/vth**2/(kappa-1.5))**(-kappa-1)*np.exp(sp.loggamma(kappa+1)-sp.loggamma(kappa-1.5))/math.pi**1.5/vth**3/(kappa-1.5)**2.5

# Calculate a normalized super-Gaussian distribution in cylindrical coordinates
# Note that we are using a loggamma approach so that we can use higher kappa values without overflow errors
def superG_norm(vperp, vpar, vth, p):
    vp = vth*(1.5*np.exp(sp.loggamma(3/p) - sp.loggamma(5/p)))**.5
    return np.exp(-(vperp**2+vpar**2)**(p/2)/vp**p)*p/4/math.pi/vp**3/sp.gamma(3/p)

