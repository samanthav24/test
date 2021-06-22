import numpy as np
import numpy.linalg as linalg
from progress.bar import Bar
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from copy import deepcopy
from init import *

"""
distance(posA, posB): distance between particle A and particle B
msd(posData): mean square displacement (dynamic property)
vsd(posData, msd): variance square displacement (dynamic property)
mnn_distance(posData): mean nearest neighbour distance (static property)
vnn_distance(posData, mnn_distance): variance nearest neighbour distance (static property)
calc_rdf(pos, pType): radial distribution function for a single timestep
"""


# def calc_distance(posA, posB):
#     """
#     calculates the distances between two particles taking the periodic boundary
#     into consideration
#     """

#     L = 28.9

#     # calculate the distance vector
#     r1 = abs(posA - posB)

#     # periodic boundary condition
#     r2 = abs(r1 - L)
#     r = np.amin([r1, r2], axis=0) 
#     r = linalg.norm(r)
#     print(r)
#     return r

def calc_distance(posA, posB):
    """
    calculates the distances between two particles taking the periodic boundary
    into consideration
    """

    L = 28.9

    rx = abs(posA[0] - posB[0])
    ry = abs(posA[1] - posB[1])
            
    # periodic boundary conditions --> rij = min( rij, abs(rij-L) )
    if rx > 0.5*L:
        rx -= L
    if ry > 0.5*L:
        ry -= L

    r = (rx**2 + ry**2)**0.5

    return r



def msd(posData):
    """
    calculates the mean square displacement for each timestep
    """

    posZero = posData[0]
    f = lambda x: linalg.norm(x - posZero, axis=2)**2
    msd = np.mean(f(posData), axis=1)

    print('Mean square displacement calculated...')
    return msd


def vsd(posData, msd):
    """
    calculates the variance square displacement for each timestep
    """

    posZero = posData[0]
    f = lambda x: linalg.norm(x-posZero, axis=2)**4
    variance = np.mean(f(posData), axis=1) - msd**2

    print('Variance square displacement calculated...')
    return variance


def mean_nn(posData, cutoff):
    """
    calculates the mean nearest neighbour distance and the mean amount
    of neighbours within the cutoff
    """
    nn_distance = np.zeros(len(posData))
    nn_amount = np.zeros(len(posData))

    bar = Bar('calc. mnn distance..', max=len(posData))

    for timestep, timestepPos in enumerate(posData):
        nn = [np.inf] * params['particles']
        nn2 = [0] * params['particles']

        # calculate the top diagonal of the distance matrix
        for i, pos in enumerate(timestepPos):
            for j, pos2 in enumerate(timestepPos[:i]):

                # calculate the distance between index1 and index2
                distance = calc_distance(pos, pos2)

                if distance < cutoff:
                    nn2[i] += 1
                    nn2[j] += 1

                if distance < nn[i]:
                    nn[i] = distance
                if distance < nn[j]:
                    nn[j] = distance

        # calculate the mean of the nearest neighbour distance
        nn_distance[timestep] = np.mean(nn)
        nn_amount[timestep] = np.mean(nn2)
        bar.next()
    
    bar.finish()
    return nn_distance, nn_amount

def variance_nn(posData, mnn_distance, mnn_amount, cutoff):
    """
    calculates the variance of the nearest neighbour distance
    """
    

    nn_distance = np.zeros(len(posData))
    nn_amount = np.zeros(len(posData))

    bar = Bar('calc. vnn distance..', max=len(posData))

    for timestep, timestepPos in enumerate(posData):
        nn = [np.inf] * params['particles']
        nn2 = [0] * params['particles']

        
        # calculate the top diagonal of the distance matrix
        for i, pos in enumerate(timestepPos):
            for j, pos2 in enumerate(timestepPos[:i]):
                
                # calculate the squared distance between particles i and j
                distance = np.square(calc_distance(pos, pos2))

                if distance < cutoff:
                    nn2[i] += 1
                    nn2[j] += 1

                if distance < nn[i]:
                    nn[i] = distance
                if distance < nn[j]:
                    nn[j] = distance

        nn_distance[timestep] = np.mean(nn) - np.square(mnn_distance[timestep])
        nn_amount[timestep] = np.mean(np.square(nn2)) - np.square(mnn_amount[timestep])

        bar.next()

    bar.finish()
    return nn_distance, nn_amount


def calc_mean(Data):
    """
    calculates the mean of the norm of the vector per timestep
    """
    f = lambda x: linalg.norm(x, axis=2)
    mean = np.mean(f(Data), axis=1)

    print('Mean calculated...')
    return mean


def calc_mean2(Data):
    """
    calculates the mean of the norm of the vector per timestep
    """
    mean = np.mean(Data, axis=1)

    print('Mean calculated...')
    return mean


def calc_variance(Data, mean):
    """
    calculates the variance of the norm of the vector per timestep
    """
    f = lambda x: np.square(linalg.norm(x, axis=2))
    variance = np.mean(f(Data), axis=1) - np.square(mean)

    print('Variance calculated...')
    return variance


def calc_variance2(Data, mean):
    """
    calculates the variance of the norm of the vector per timestep
    """
    f = lambda x: np.square(x)
    variance = np.mean(f(Data), axis=1) - np.square(mean)

    print('Variance calculated...')
    return variance


def calc_rdf(pos, pType):
    """
    calculates the radial distribution function
    @param :pos: position of all the particles in a single timestep
    @param :pType: list of particle types, 1 or 2
    """

    dr        = params['dr']
    rmax      = params['rmax']
    A         = params['A']
    Lx        = params['Lx']
    Ly        = params['Ly']
    r         = np.arange(0,rmax+dr,dr)
    NR        = len(r)
    grAA      = np.zeros((NR, 1))
    grBB      = np.zeros((NR, 1))
    grAB      = np.zeros((NR, 1))

    iA = np.where(pType==1)[0]
    iB = np.where(pType==2)[0]
    NA = len(iA)
    NB = len(iB)

    for i in range(params['particles']):
        for j in range(i+1, params['particles']):
            
            rx = abs(pos[i, 0] - pos[j, 0])
            ry = abs(pos[i, 1] - pos[j, 1])
                 
            # periodic boundary conditions --> rij = min( rij, abs(rij-L) )
            if rx > 0.5*Lx:
                rx -= Lx
            if ry > 0.5*Ly:
                ry -= Ly

            r = np.sqrt(rx**2 + ry**2)

            if r <= rmax:
                igr = round(r/dr)

                if i in iA and j in iA:
                    grAA[igr] = grAA[igr] + 2
                elif i in iB and j in iB:
                    grBB[igr] = grBB[igr] + 2
                else:
                    grAB[igr] = grAB[igr] + 1

    # normalize
    dr2       = np.zeros((NR,1))

    for ir in range(NR):
        rlow    = ir*dr
        rup     = rlow + dr
        dr2[ir] = rup**2 - rlow**2

    nidealA   = np.pi * dr2 * (NA*NA/A)            # g(r) for ideal gas of A particles
    nidealB   = np.pi * dr2 * (NB*NB/A)            # g(r) for ideal gas of B particles
    nidealAB  = np.pi * dr2 * (NA*NB/A)            # g(r) for ideal gas of A+B particles

    grAA_norm = grAA / nidealA
    grBB_norm = grBB / nidealB
    grAB_norm = grAB / nidealAB
    r         = np.arange(0,rmax+dr,dr)


    return grAA_norm, grBB_norm, grAB_norm

def calc_rdf_peaks(posData, types):
    """
    calculates the radial distribution function peak for AA, AB and BB.
    Only for AA and AB does this value also correspond to the first peak
    """

    grAA_amax, grBB_amax, grAB_amax = np.zeros(len(posData)), np.zeros(len(posData)), np.zeros(len(posData))

    bar = Bar('calc. rdf peaks..', max=len(posData))

    for t, pos_t, type_t in zip(range(len(posData)), posData, types):

        grAA, grBB, grAB = calc_rdf(pos_t, type_t)

        # calculate the argmax corresponding to the first peak for grAA and grAB
        grAA_amax[t] = np.argmax(grAA)
        grBB_amax[t] = np.argmax(grBB)
        grAB_amax[t] = np.argmax(grAB)

        bar.next()

    bar.finish()
    
    return grAA_amax, grBB_amax, grAB_amax

def calc_rdf_minimum(posData, types):
    """
    calculates the radial distribution function minimum for AA, AB and BB.
    """

    grAA_amin, grBB_amin, grAB_amin = np.zeros(len(posData)), np.zeros(len(posData)), np.zeros(len(posData))

    bar = Bar('calc. rdf minimum..', max=len(posData))

    for t, pos_t, type_t in zip(range(len(posData)), posData, types):

        grAA, grBB, grAB = calc_rdf(pos_t, type_t)

        # calculate the argmax corresponding to the first peak for grAA and grAB
        grAA_amax = np.argmax(grAA)
        grBB_amax = np.argmax(grBB)
        grAB_amax = np.argmax(grAB)

        # reload the arrays but from the maximum index and further
        grAA = grAA[grAA_amax:]
        grBB = grBB[grBB_amax:]
        grAB = grBB[grAB_amax:]

        # calculate the argmin corresponding to the first minimum for grAA, grAB, grBB
        grAA_amin[t] = np.argmin(grAA) + grAA_amax
        grBB_amin[t] = np.argmin(grBB) + grBB_amax
        grAB_amin[t] = np.argmin(grAB) + grAB_amax

        bar.next()

    bar.finish()
    
    return grAA_amin, grBB_amin, grAB_amin


def calc_rdf_area(posData, types):
    """
    calculates the area under the radial distribution function for AA, AB and BB.
    """

    dr = params['dr']
    grAA_area, grBB_area, grAB_area = np.zeros(len(posData)), np.zeros(len(posData)), np.zeros(len(posData))

    bar = Bar('calc. rdf area..', max=len(posData))

    for t, pos_t, type_t in zip(range(len(posData)), posData, types):

        grAA, grBB, grAB = calc_rdf(pos_t, type_t)

        # calculate the area under the rdf graph per timestep
        grAA_area[t] = np.sum(grAA * dr)
        grBB_area[t] = np.sum(grBB * dr)
        grAB_area[t] = np.sum(grAB * dr)

        bar.next()

    bar.finish()

    return grAA_area, grBB_area, grAB_area

def calc_avg_rdf(posData, types):
    """
    calculates the average rdf over all the timesteps
    """

    dr        = params['dr']
    rmax      = params['rmax']
    r         = np.arange(0,rmax+dr,dr)
    NR        = len(r)
    grAA      = np.zeros((NR, 1))
    grBB      = np.zeros((NR, 1))
    grAB      = np.zeros((NR, 1))

    bar = Bar('calc. rdf peaks..', max=len(posData))

    for pos_t, type_t in zip(posData, types):

        grAAt, grBBt, grABt = calc_rdf(pos_t, type_t)
        grAA += grAAt
        grBB += grBBt
        grAB += grABt

        bar.next()

    bar.finish()

    grAA /= len(posData)    
    grBB /= len(posData)
    grAB /= len(posData)

    return grAA, grBB, grAB


def plot_rdf(r, gr, title):
    """
    plots a radial distribution function
    """
    plt.plot(r, gr)
    plt.title(title)
    plt.xlabel('r')
    plt.ylabel('gr')
    plt.show()


def calc_rdf_all(posData, types):
    """
    calculates the radial distribution function peak for AA, AB and BB for
    all timesteps
    """
    r = np.arange(0,params['rmax']+params['dr'],params['dr'])
    grAA, grBB, grAB = np.zeros((len(posData), len(r), 1)), np.zeros((len(posData), len(r), 1)), np.zeros((len(posData), len(r), 1))

    bar = Bar('calc. rdf peaks..', max=len(posData))

    for t, pos_t, type_t in zip(range(len(posData)), posData, types):

        grAA_t, grBB_t, grAB_t = calc_rdf(pos_t, type_t)

        # calculate the argmax corresponding to the first peak for grAA and grAB
        grAA[t] = grAA_t
        grBB[t] = grBB_t
        grAB[t] = grAB_t

        bar.next()

    bar.finish()    

    return grAA, grBB, grAB


def rdf_poi(rdf):
    """
    calculates all the points of interest for the radial distribution function 
    """

    features = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    for t in range(len(rdf[0])):
        grAA, grBB, grAB = rdf[0][t], rdf [1][t], rdf[2][t]

        # first peak of grAA
        features[0].append(max(grAA))

        # second peak of grAA
        left = round(1.5 / params['dr'])
        right = round(2 / params['dr'])
        features[1].append(max(grAA[left:right]))  

        # third peak of grAA
        left = round(2 / params['dr'])
        right = round(2.5 / params['dr'])
        features[2].append(max(grAA[left:right])) 

        # fourth peak of grAA
        left = round(2.5 / params['dr'])
        right = round(3 / params['dr'])
        features[3].append(max(grAA[left:right]))  

        # first minimum of grAA
        left = round(1.2 / params['dr'])
        right = round(1.75 / params['dr'])
        features[4].append(min(grAA[left:right]))  

        # second minimum of grAA
        left = round(1.8 / params['dr'])
        right = round(2.1 / params['dr'])
        features[5].append(min(grAA[left:right]))  

        # third minimum of grAA
        left = round(2.1 / params['dr'])
        right = round(2.7 / params['dr'])
        features[6].append(min(grAA[left:right]))  

        # first peak of grAB
        features[7].append(max(grAB))

        # second peak of grAB
        left = round(1.5 / params['dr'])
        right = round(2.2 / params['dr'])
        features[8].append(max(grAB[left:right]))  

        # first minimum of grAB
        left = round(1 / params['dr'])
        right = round(1.7 / params['dr'])
        features[9].append(min(grAB[left:right])) 

        # second minimum of grAB
        left = round(2 / params['dr'])
        right = round(2.5 / params['dr'])
        features[10].append(min(grAB[left:right])) 

        # first peak of grBB
        features[11].append(max(grBB))

        # second peak of grBB
        left = round(2.5 / params['dr'])
        right = round(3 / params['dr'])
        features[12].append(max(grBB[left:right])) 

        # first minimum of grBB
        left = round(1.8 / params['dr'])
        right = round(2.4 / params['dr'])
        features[13].append(min(grBB[left:right])) 

        # second minimum of grBB
        left = round(2.8 / params['dr'])
        right = round(3.3 / params['dr'])
        features[14].append(min(grBB[left:right])) 

    return features

def calc_voronoi(timesteps, types, posData):
    edgesA = []
    edgesB = []
    area_peakA_count, area_peakB_count = np.zeros(len(timesteps)), np.zeros(len(timesteps))
    area_peakA_mag, area_peakB_mag = np.zeros(len(timesteps)), np.zeros(len(timesteps))

    # index = 0
    for timestep in timesteps:

        # periodic boundary conditions
        pos = posData[timestep]
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
        edgesA.append(edgesAmount[types[timestep] == 1])
        edgesB.append(edgesAmount[types[timestep] == 2])

        # # get voronoi area
        # area = np.zeros(params['particles'])
        # for i, reg_num in enumerate(vor.point_region[:1000]):
        #     indices = vor.regions[reg_num]
        #     if -1 in indices: # some regions can be opened
        #         area[i] = np.inf
        #     else:
        #         area[i] = ConvexHull(vor.vertices[indices]).volume # volume corresponds to area for 2d data

        # # get peak voronoi area A particles
        # areaA = area[types[timestep] == 1]
        # values, bins, _ = plt.hist(areaA, bins=10)
        # order = np.argsort(values)[::-1]

        # area_peakA_count_t = values[order][:1]
        # area_peakA_mag_t = [bins[i] + (bins[i+1] - bins[i])/2 for i in order[:1]]

        # area_peakA_count_t, area_peakA_mag_t = np.array(area_peakA_count_t, dtype= 'float'), np.array(area_peakA_mag_t, dtype= 'float')

        # area_peakA_count[index] = area_peakA_count_t
        # area_peakA_mag[index] = area_peakA_mag_t

        # # get peak voronoi area B particles
        # areaB = area[types[timestep] == 2]

        # values, bins, _ = plt.hist(areaB, bins=10)
        # order = np.argsort(values)[::-1]

        # area_peakB_count_t = values[order][:1]
        # area_peakB_mag_t = [bins[i] + (bins[i+1] - bins[i])/2 for i in order[:1]]

        # area_peakB_count_t, area_peakB_mag_t = np.array(area_peakB_count_t, dtype= 'float'), np.array(area_peakB_mag_t, dtype= 'float')

        # area_peakB_count[index] = area_peakB_count_t
        # area_peakB_mag[index] = area_peakB_mag_t
        
        # index += 1

    # get count voronoi edges
    edgesA5, edgesA6, edgesA7 = [], [], []
    for edgeA in edgesA:
        edgesA5.append(np.count_nonzero(edgeA == 5))
        edgesA6.append(np.count_nonzero(edgeA == 6))
        edgesA7.append(np.count_nonzero(edgeA == 7))

    edgesB5, edgesB6, edgesB7 = [], [], []
    for edgeB in edgesB:
        edgesB5.append(np.count_nonzero(edgeB == 5))
        edgesB6.append(np.count_nonzero(edgeB == 6))
        edgesB7.append(np.count_nonzero(edgeB == 7))

    return edgesA5, edgesA6, edgesA7, edgesB5, edgesB6, edgesB7

def save_load(func, savename):
    """
    tries to load the given funcion from npy files, if it is not there
    run the function
    """
    try:
        with open(savename, 'rb') as f:
            result = np.load(f)
            print('Feature loaded..')
    except:
        result = np.asarray(func())
        with open(savename, 'wb') as f:
            np.save(f, result)

    return result


def calc_SF(rdf):
    """
    calculates the structure factor by taking the fourier transform of the rdf
    @param :rdf: radial distribution function of a single timestep
    @returns : SF_AA, SF_BB, SF_AB
    """
    dr = params['dr']
    rmax = params['rmax']
    density = params['density']
    SF_AA, SF_BB, SF_AB = [], [], []

    for q in np.arange(dr, rmax, dr):
        sum_AA, sum_BB, sum_AB = 0, 0, 0
        incre = 0
        
        for r in np.arange(dr, rmax, dr):
            incre += 1
            sum_AA += r * (rdf[0][incre][0] - 1) * np.sin(q * r)
            sum_BB += r * (rdf[1][incre][0] - 1) * np.sin(q * r)
            sum_AB += r * (rdf[2][incre][0] - 1) * np.sin(q * r)

        SF_AA.append(1 + 4 * np.pi * density * sum_AA * dr / q)
        SF_BB.append(1 + 4 * np.pi * density * sum_BB * dr / q)
        SF_AB.append(1 + 4 * np.pi * density * sum_AB * dr / q)
    
    return SF_AA, SF_BB, SF_AB