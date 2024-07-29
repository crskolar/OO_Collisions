# Import packages
import numpy as np
import string

# This function find where the white space is in a line of strings. Used for parsing fortran data
def find_whitespace(st):
    for index, character in enumerate(st):
       if character in string.whitespace:
            yield index

# This function reads a DMP file from the fortran output to the Monte Carlo code
def readDMP(fileName):
    DMPfile = open(fileName)
    lines = DMPfile.readlines()
    
    # We know that this should end up as a 2D array with 24x24 elements
    output = np.zeros((24,24))
    
    counter_row = 0
    counter_col = 0
    # Strips the newline character
    for line in lines:
        # If the line is not a new line, then it is a line that has data in it that we want to get
        if line != '\n':
            # Find where all the spaces are in the file. Note that there are two spaces in between every data point
            whitespace_indices = list(find_whitespace(line))
    
            # Get the number of data points in the line. If counter_row is 4, then the number of points in the line is 4. Otherwise, it is 5
            if counter_col == 20:
                numPoints_in_line = 4
            else:
                numPoints_in_line = 5
                
            # Iterate through the number of points in the line
            for i in range(0,numPoints_in_line):
                # Place the point in the appropriate spot in the 2D array
                output[counter_row,counter_col] = np.double(line[whitespace_indices[2*i+1]+1:whitespace_indices[2*i+2] ])
                counter_col += 1
    
            if counter_col == 24:
                counter_col = 0
                counter_row += 1
    # So output is the one quadrant distribution function
    dv = 1.7e3
    vperp = np.arange(0,24*dv,dv)
    vpar = np.arange(-23*dv,24*dv,dv)
    [VVperp,VVpar] = np.meshgrid(vperp, vpar)
    
    f0Full = np.zeros((47,24))
    f0Full[23:,:] = output
    f0Full[0:23,:] = np.flip(output[1:,:],0)
    
    return output, vperp, vpar, VVperp, VVpar, f0Full


# This function reads a DMPV file from the Monte Carlo output
def readDMPV(fileName):
    # Load the data skipping the first row
    data = np.loadtxt(fileName,skiprows=1)

    # Iterate through the total number of elements in data
    # Start at 1. 
    for i in range(0,len(data)-1):
        # Iterate through the initial few values of vperp
        if data[i,1] != data[i+1,1]: 
            # When the value of vperp is not the same as next value, then we know what the length of the vparallel array is
            numPar = int(i+1)
            break

    # Get the length of vperp
    numPerp = int(len(data)/numPar)

    # Pre-allocate arrays for the velocity meshes and the distribution function
    VVperp = np.zeros((numPar, numPerp))
    VVpar = np.zeros((numPar, numPerp))
    f0 = np.zeros((numPar, numPerp))
    extra = np.zeros((numPar, numPerp))

    counter = 0
    for i in range(numPar):
        VVpar[:,i] = data[i*numPerp:(i+1)*numPerp,0]
        VVperp[:,i] = data[i*numPerp:(i+1)*numPerp,1]
        f0[:,i] = data[i*numPerp:(i+1)*numPerp,2]
        extra[:,i] = data[i*numPerp:(i+1)*numPerp,3]
        
    # Input velocities are in km/s. Convert to m/s and return 
    return VVperp*1000, VVpar*1000, f0, extra