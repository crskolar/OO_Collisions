import numpy as np


fileDir = 'C:/Users/Chirag/Documents/O+O/data/interp_speed/'

avg_time_per_calc = np.zeros(10)

for numProc in range(1,11):
    fileName = 'par2_perp1_theta50_np' + str(numProc) + '.log'
    # nmax = 2*numProc-1
    nmax = 10*numProc-1
    local_times = np.zeros(nmax+1)

    lineNum = 0 
    n = 0
    with open(fileDir + fileName) as file:
        for line in file:
            lineNum += 1
            if lineNum != 1 and line.rstrip()[-9] != "F":
                local_times[n] = float(line.rstrip()[-9:-1])
                n += 1
                
    avg_time_per_calc[numProc-1] = np.sum(local_times)/(2*nmax+1)
    time_per_proc = avg_time_per_calc/np.arange(1,11)
    