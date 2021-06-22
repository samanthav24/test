from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from data_process import *
from init import *
import os
from progress.counter import Counter
import sys
import numpy as np

def avg_voronoi(timestep):
    """
    Returns the voronoi area and the amount of voronoi edges for each particle 
    averaged over all dump files for given timestep
    """

    # initialization
    if timestep >= 0 and timestep <= 50000:
        iterations = 50000
        dump_interval = 50
        directory = ".\\dump\\young\\"
        extension = ".YOUNG"
    elif timestep >= 4000000 and timestep <= 5000000:
        timestep -= 4000000
        iterations = 1000000
        dump_interval = 1000
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

            print("Calculating file " + str(amount) + ' for timestep ' + str(timestep * dump_interval), sep=' ', end='\r', file=sys.stdout, flush=False)

            _, types, _, _, vor_area, vor_amn, Data = read_data(directory + filename, iterations, dump_interval)

            if amount == 0:
                voronoi_area = vor_area[timestep]
                voronoi_amount = vor_amn[timestep]
            else:
                voronoi_area += vor_area[timestep]
                voronoi_amount += vor_amn[timestep]
            
            amount += 1

    voronoi_area /= amount
    voronoi_amount /= amount

    return voronoi_area, voronoi_amount

def plot_voronoi(timestep_list):
    """
    Plots histograms as line graphs of the voronoi area and the amount of voronoi edges
    for given timesteps (averaged over all dump files)
    """
    
    voronoi_area_list = []
    voronoi_amount_list = []
    for timestep in timestep_list:
        voronoi_area, voronoi_amount = avg_voronoi(timestep)
        voronoi_area_list.append(voronoi_area)
        voronoi_amount_list.append(voronoi_amount)

    i = 0
    for timestep in timestep_list:
        
        voronoi_area = voronoi_area_list[i]
        values, bins = np.histogram(voronoi_area, bins=20)
        bin_centers = 0.5*(bins[1:]+bins[:-1])

        plot_label = "t =" + str(timestep)
        plt.plot(bin_centers, values, label=plot_label)
        i += 1

    plt.legend()
    plt.xlabel('Voronoi area')
    plt.ylabel('frequency')
    plt.title("Histogram of Voronoi area")
    plt.show()

    i = 0
    for timestep in timestep_list:
        
        voronoi_amount = voronoi_amount_list[i]
        values, bins = np.histogram(voronoi_amount, bins=10)
        bin_centers = 0.5*(bins[1:]+bins[:-1])

        plot_label = "t =" + str(timestep)
        plt.plot(bin_centers, values, label=plot_label)
        i += 1

    plt.legend()
    plt.xlabel('Voronoi edges')
    plt.ylabel('frequency')
    plt.title("Histogram of amount of Voronoi edges")
    plt.show()

plot_voronoi([100, 1000, 4000000, 5000000])