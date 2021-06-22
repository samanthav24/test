from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from data_process import *
from init import *
import os
import sys
from scipy.spatial import Voronoi, voronoi_plot_2d
from copy import deepcopy



def voronoi(timesteps, full=False):
    """
    calculates the average feature of all available files for a particular timestep
    @param :timestep: timestep of interest
    """

    timesteps = np.asarray(timesteps)

    if full:
        iterations = [5000000]
        dump_interval = [1000]
        directory = [".\\dump\\full\\"]
        extension = [".ATOM"]
        timesteps = [timesteps]
    else:
        iterations = [50000, 1000000]
        dump_interval = [50, 1000]
        directory = [".\\dump\\young\\", ".\\dump\\old\\"]
        extension = [".YOUNG", ".OLD"]
        youngamount = len(timesteps[timesteps <= 50000])
        timesteps = [timesteps[timesteps <= 50000], timesteps[timesteps >= 4000000]]

    areasA = []
    areasB = []
    edges = []
    edgesA = []
    edgesB = []

    for iter, dump, dir, ext, time in zip(iterations, dump_interval, directory, extension, timesteps):
        
        files = 0

        # iterate through dump files
        for file in os.listdir(os.fsencode(dir)):
            filename = os.fsdecode(file)
            if filename.endswith(ext): 
                
                t = 0
                _, types, _, _, _, _, _, Data = read_data(dir + filename, iter, dump)

                for timestep in time:

                    print("Calculating file " + str(files) + ' for timestep ' + str(timestep), sep=' ', end='\r', file=sys.stdout, flush=False)
                    
                    if ext == '.OLD':
                        timestep -= 4000000
                        index = t + youngamount
                    else:
                        index = t

                    timestep = int(timestep / dump)

                    # periodic boundary conditions
                    pos = Data[0, timestep]
                    oldpos = deepcopy(pos)
                    pos2 = pos - 28.9
                    pos3 = pos + 28.9
                    pos5 = deepcopy(pos)
                    pos5[:, 0] += 28.9
                    pos6 = deepcopy(pos)
                    pos6[:, 0] -= 28.9
                    pos7 = deepcopy(pos)
                    pos7[:, 1] += 28.9
                    pos8 = deepcopy(pos)
                    pos8[:, 1] -= 28.9
                    pos9 = deepcopy(pos)
                    pos9[:, 0] += 28.9
                    pos9[:, 1] -= 28.9
                    pos10 = deepcopy(pos)
                    pos10[:, 1] += 28.9
                    pos10[:, 0] -= 28.9                    
                    posNew = deepcopy(pos)
                    pos = np.concatenate((pos, pos2, pos3, pos5, pos6, pos7, pos8, pos9, pos10), axis=0)
                    
                    vor = Voronoi(pos)
                    regions = np.array(vor.regions, dtype=object)

                    # get amount of voronoi edges
                    edgesAmount = regions[(vor.point_region[:1000])]
                    edgesAmount = np.array([len(x) for x in edgesAmount])

                    # get voronoi area
                    area = np.zeros(params['particles'])
                    for i, reg_num in enumerate(vor.point_region[:1000]):
                        indices = vor.regions[reg_num]
                        if -1 in indices: # some regions can be opened
                            area[i] = np.inf
                        else:
                            area[i] = ConvexHull(vor.vertices[indices]).volume # volume corresponds to area for 2d data
                    
                    area = np.array(area)

                    if files == 0:
                        edges.append(edgesAmount)
                        edgesA.append(edgesAmount[types[timestep] == 1])
                        edgesB.append(edgesAmount[types[timestep] == 2])
                        areasA.append(area[types[timestep] == 1])
                        areasB.append(area[types[timestep] == 2])
                    else:
                        edges[index] = np.concatenate((edges[index], edgesAmount))
                        edgesA[index] = np.concatenate((edgesA[index], edgesAmount[types[timestep] == 1]))
                        edgesB[index] = np.concatenate((edgesB[index], edgesAmount[types[timestep] == 2]))
                        areasA[index] = np.concatenate((areasA[index], area[types[timestep] == 1]))
                        areasB[index] = np.concatenate((areasB[index], area[types[timestep] == 2]))
                    

                    t += 1

                files += 1

    edges, edgesA, edgesB = np.array(edges, dtype= 'float'), np.array(edgesA, dtype= 'float'), np.array(edgesB, dtype= 'float')
    areasA, areasB = np.array(areasA, dtype= 'float'), np.array(areasB, dtype= 'float')

    return edges, edgesA, edgesB, areasA, areasB, files


def plot_voronoi(timesteps, distinct=False):
    """
    Plots histograms as line graphs of  the amount of voronoi edges
    for given timesteps (averaged over all dump files)
    @param :timesteps: list of timesteps of interest
           :distinct: boolean, True if you want a distinction between 
                      A and B particles in the plot
    """

    edges, edgesA, edgesB, areasA, areasB, files = voronoi(timesteps)

    if distinct is True:

        # vorornoi edges
        for timestep, edgeA, edgeB in zip(timesteps, edgesA, edgesB):

            # get frequencies and average over files
            values_A, bins_A = np.histogram(edgeA, bins=[3, 4, 5, 6, 7, 8, 9, 10])
            values_A = np.array(values_A, dtype= 'float')
            values_A /= files
            bins_A = bins_A[:-1]
            plt.plot(bins_A, values_A, label='A: t =' + str(timestep))

            # get frequencies and average over files
            values_B, bins_B = np.histogram(edgeB, bins=[3, 4, 5, 6, 7, 8, 9, 10])
            values_B = np.array(values_B, dtype= 'float')
            values_B /= files
            bins_B = bins_B[:-1]
            plt.plot(bins_B, values_B, label='B: t =' + str(timestep))

        plt.legend()
        plt.title('Histogram of amount of Voronoi edges')
        plt.xlabel('Voronoi edges')
        plt.ylabel('Frequency')
        plt.show()

        # voronoi area
        for timestep, areaA, areaB in zip(timesteps, areasA, areasB):
 
            # get frequencies and average over files
            values_A, bins_A = np.histogram(areaA, bins=10)
            values_A = np.array(values_A, dtype= 'float')
            values_A /= files
            bins_A = bins_A[:-1]
            plt.plot(bins_A, values_A, label='A: t =' + str(timestep))

            # get frequencies and average over files
            values_B, bins_B = np.histogram(areaB, bins=10)
            values_B = np.array(values_B, dtype= 'float')
            values_B /= files
            bins_B = bins_B[:-1]
            plt.plot(bins_B, values_B, label='B: t =' + str(timestep))

        plt.legend()
        plt.title('Histogram of Voronoi area')
        plt.xlabel('Voronoi area')
        plt.ylabel('Frequency')
        plt.show()
    
    else:
        for timestep, edge in zip(timesteps, edges):

            # get frequencies and average over files
            values, bins = np.histogram(edge, bins=[3, 4, 5, 6, 7, 8, 9, 10])
            values = np.array(values, dtype= 'float')
            values /= files
            bins = bins[:-1]
            plt.plot(bins, values, label='t =' + str(timestep))

        plt.legend()
        plt.title('Histogram of amount of Voronoi edges')
        plt.xlabel('Voronoi edges')
        plt.ylabel('Frequency')
        plt.show()

timesteps = [100, 10000, 4000000, 5000000]
plot_voronoi(timesteps, distinct=True)