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

def avg_voronoi_AB(timestep):
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

    index_A = np.where(types[timestep] == 1)
    index_B = np.where(types[timestep] == 2)

    voronoi_area_A = np.take(voronoi_area, index_A)
    voronoi_area_B = np.take(voronoi_area, index_B)
    voronoi_amount_A = np.take(voronoi_amount, index_A)
    voronoi_amount_B = np.take(voronoi_amount, index_B)

    return voronoi_area_A, voronoi_amount_A, voronoi_area_B, voronoi_amount_B

def plot_voronoi_AB(timestep_list):
    """
    Plots histograms as line graphs of the voronoi area and the amount of voronoi edges
    for given timesteps (averaged over all dump files)
    """
    
    voronoi_area_list_A, voronoi_amount_list_A = [], []
    voronoi_area_list_B, voronoi_amount_list_B = [], []

    for timestep in timestep_list:
        voronoi_area_A, voronoi_amount_A, voronoi_area_B, voronoi_amount_B = avg_voronoi_AB(timestep)

        voronoi_area_list_A.append(voronoi_area_A)
        voronoi_amount_list_A.append(voronoi_amount_A)
        voronoi_area_list_B.append(voronoi_area_B)
        voronoi_amount_list_B.append(voronoi_amount_B)

    i = 0
    for timestep in timestep_list:
        
        voronoi_area_A = voronoi_area_list_A[i]
        voronoi_area_B = voronoi_area_list_B[i]
        min_value = voronoi_area_B.min()
        max_value = voronoi_area_A.max()

        values_A, bins_A = np.histogram(voronoi_area_A, bins=20, range=(min_value, max_value))
        bin_centers_A = 0.5*(bins_A[1:]+bins_A[:-1])
        plot_label = "A: t =" + str(timestep)
        plt.plot(bin_centers_A, values_A, label=plot_label)

        values_B, bins_B = np.histogram(voronoi_area_B, bins=20, range=(min_value, max_value))
        bin_centers_B = 0.5*(bins_B[1:]+bins_B[:-1])
        plot_label = "B: t =" + str(timestep)
        plt.plot(bin_centers_B, values_B, label=plot_label)

        i += 1

    plt.legend()
    plt.xlabel('Voronoi area')
    plt.ylabel('frequency')
    plt.title("Histogram of Voronoi area")
    plt.show()

    i = 0
    for timestep in timestep_list:
        
        voronoi_amount_A = voronoi_amount_list_A[i]
        voronoi_amount_B = voronoi_amount_list_B[i]
        min_value = voronoi_amount_B.min()
        max_value = voronoi_amount_A.max()

        values_A, bins_A = np.histogram(voronoi_amount_A, bins=10, range=(min_value, max_value))
        bin_centers_A = 0.5*(bins_A[1:]+bins_A[:-1])
        plot_label = "A: t =" + str(timestep)
        plt.plot(bin_centers_A, values_A, label=plot_label)

        values_B, bins_B = np.histogram(voronoi_amount_B, bins=10, range=(min_value, max_value))
        bin_centers_B = 0.5*(bins_B[1:]+bins_B[:-1])
        plot_label = "B: t =" + str(timestep)
        plt.plot(bin_centers_B, values_B, label=plot_label)

        i += 1

    plt.legend()
    plt.xlabel('Voronoi edges')
    plt.ylabel('frequency')
    plt.title("Histogram of amount of Voronoi edges")
    plt.show()

plot_voronoi_AB([100, 1000])