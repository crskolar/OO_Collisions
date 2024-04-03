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

real_z = 100
imag_z = 1e-6

num_points = 100
dg = 100*imag_z

grid_base = np.logspace(0,np.log10(dg+1),num_points)

diff =  grid_base - 1

grid = np.unique(np.concatenate(( -diff, diff)) + real_z)

# plt.plot(diff, np.zeros(num_points),'.')

plt.plot(grid, np.zeros_like(grid),'.')
# plt.plot(-diff, np.zeros(num_points),'.')
plt.grid()

for i in range(1,len(grid)):
    print(grid[i] - grid[i-1])
    
    