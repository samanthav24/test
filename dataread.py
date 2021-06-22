import numpy as np
import pandas as pd
from init import *


def read_data(fileName, iterations, dump_interval):
    """
    reads a .atom file and returns the particles 3 dimensional position
    per timestep in a numpy array
    """

    size = int((iterations / dump_interval) + 1)

    # array of timesteps in which the position and forces of all atoms can be found
    posData = np.zeros((size, params['particles'], params['dimensions']))
    forceData = np.zeros((size, params['particles'], params['dimensions']))
    q4 = np.zeros((size, params['particles'], 2))
    q5 = np.zeros((size, params['particles'], 2))
    q6 = np.zeros((size, params['particles'], 2))
    q7 = np.zeros((size, params['particles'], 2))
    q8 = np.zeros((size, params['particles'], 2))

    timesteps = np.arange(0, size)
    types = np.zeros((size, params['particles']))


    f = open(fileName, "r")
    i = 0

    for line in f:

        # new timestep
        if 'ITEM: TIMESTEP' in line:
            i = 0

        # get the timestep value
        if i == 1:
            timestep = int(int(line) / dump_interval)

        # skip first 8 lines for each timestep
        if i > 8:

            line = line.split(' ')

            particleId = int(line[0]) - 1
            types[timestep, particleId] = int(line[1])

            # position, force, dipole moment orientation, dipole moment magnitude, charge, angular momentum, torque
            position = np.array([line[2 + i] for i in range(params['dimensions'])])
            force = np.array([line[2 + params['dimensions'] + i] for i in range(params['dimensions'])])
            q4_value = np.array([line[6], line[7]])
            q5_value = np.array([line[8], line[9]])
            q6_value = np.array([line[10], line[11]])
            q7_value = np.array([line[12], line[13]])
            q8_value = np.array([line[14], line[15]])

            posData[timestep, particleId] = position
            forceData[timestep, particleId] = force
            q4[timestep, particleId] = q4_value
            q5[timestep, particleId] = q5_value
            q6[timestep, particleId] = q6_value
            q7[timestep, particleId] = q7_value
            q8[timestep, particleId] = q8_value

        i += 1

    return timesteps, types, q4, q5, q6, q7, q8, np.array([posData, forceData])
