from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from data_process import *
from init import *
import os
import sys


def avg_feature(timestep):
    """
    calculates the average feature of all available files for a particular timestep
    @param :timestep: timestep of interest
    """

    # initialization
    if timestep >= 0 and timestep <= 50000:
        iterations = 50000
        dump_interval = 50
        directory = ".\\dump\\young\\"
        extension = ".YOUNG"
    elif timestep >= 4000000 and timestep <= 5000000:
        iterations = 1000000
        dump_interval = 1000
        timestep -= 4000000
        directory = ".\\dump\\old\\"
        extension = ".OLD"
    else:
        raise ValueError("No data available for this timestep")

    if timestep % dump_interval != 0:
        raise ValueError("Timestep has to be dividible by the dump_interval")

    timestep = int(timestep / dump_interval)
    amount = 0

    # iterate through dump files
    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(extension): 
            
            print("Calculating file " + str(amount) + ' for timestep ' + str(timestep), sep=' ', end='\r', file=sys.stdout, flush=False)

            _, types, _, _, _, _, Data = read_data(directory + filename, iterations, dump_interval)
            
            if amount == 0:
                rdf = np.asarray(calc_rdf(Data[0, timestep], types[timestep]))
                SF = np.asarray(calc_SF(rdf))
            else:
                rdf_new = np.asarray(calc_rdf(Data[0, timestep], types[timestep]))
                rdf += rdf_new
                SF += np.asarray(calc_SF(rdf))
            
            amount += 1
    
    rdf /= amount
    SF /= amount
    return rdf, SF

rdf = []
SF = []
timesteps = [0, 50000, 4000000, 5000000]

for timestep in timesteps:

    rdf_new, SF_new = avg_feature(timestep)
    rdf.append(rdf_new)
    SF.append(SF_new)

r = np.arange(0,params['rmax'] + params['dr'], params['dr'])
q = r

rdf_mean = np.sum(rdf, axis=0) / len(timesteps)
SF_mean = np.sum(SF, axis=0) / len(timesteps)

# plot the rdf and the structure factor
for i, title in zip([0, 1, 2], ['grAA', 'grBB', 'grAB']):
    for timestep, j in zip(timesteps, np.arange(len(timesteps))):
        plt.plot(r, rdf[j][i], label='t = ' + str(timestep))

    plt.legend()
    plt.title(title)
    plt.ylabel('g(r)')
    plt.xlabel('r')
    plt.show()

for i, title in zip([0, 1, 2], ['SF_AA', 'SF_BB', 'SF_AB']):
    for timestep, j in zip(timesteps, np.arange(len(timesteps))):
        plt.plot(q[1:], SF[j][i], label='t = ' + str(timestep))
    plt.legend()
    plt.title(title)
    plt.ylabel('SF(q)')
    plt.xlabel('q')
    plt.show()

rdf -= rdf_mean
SF -= SF_mean

# plot the rdf and the structure factor
for i, title in zip([0, 1, 2], ['grAA', 'grBB', 'grAB']):
    for timestep, j in zip(timesteps, np.arange(len(timesteps))):
        plt.plot(r, rdf[j][i], label='t = ' + str(timestep))
    plt.legend()
    plt.title(title)
    plt.ylabel('g(r)')
    plt.xlabel('r')
    plt.show()

for i, title in zip([0, 1, 2], ['SF_AA', 'SF_BB', 'SF_AB']):
    for timestep, j in zip(timesteps, np.arange(len(timesteps))):
        plt.plot(q[1:], SF[j][i], label='t = ' + str(timestep))
    plt.legend()
    plt.title(title)
    plt.ylabel('SF(q)')
    plt.xlabel('q')
    plt.show()
