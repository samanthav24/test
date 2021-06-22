# calculates cutoff at first minimum
from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from data_process import *
from init import *
import os

def calc_rdf_cutoff(age):

    dr = params['dr']

    if age == 'young':
        directory = ".\\dump\\young\\"
        file_ending = ".YOUNG"
        iterations = 50000 
        dump_interval = 50

    if age == 'old':
        directory = ".\\dump\\old\\"
        file_ending = ".OLD"
        iterations = 1000000 
        dump_interval = 1000   

    grAA, grBB, grAB = [], [], []

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(file_ending):

            print('calculating', filename)

            # load data
            timesteps, types, q6_re, q6_im, Data = read_data(directory + filename, iterations, dump_interval)
            posData = Data[0]

            # get minimum for each timestep
            grAA_amin, grBB_amin, grAB_amin = calc_rdf_minimum(posData, types)

            print(grAA_amin.mean())

            # get mean of cutoff for different times
            grAA.append(grAA_amin.mean())
            grBB.append(grBB_amin.mean())
            grAB.append(grAB_amin.mean())

    grAA = np.array(grAA)
    grBB = np.array(grBB)
    grAB = np.array(grAB)

    # get mean of cutoff for different files
    AA_cutoff = np.mean(grAA) * dr
    BB_cutoff = np.mean(grBB) * dr
    AB_cutoff = np.mean(grAB) * dr

    return AA_cutoff, BB_cutoff, AB_cutoff

AA_cutoff, BB_cutoff, AB_cutoff = calc_rdf_cutoff('young')
print('AA_cutoff young:', AA_cutoff)
print('BB_cutoff young:', BB_cutoff)
print('AB_cutoff young:', AB_cutoff)

AA_cutoff, BB_cutoff, AB_cutoff = calc_rdf_cutoff('old')
print('AA_cutoff old:', AA_cutoff)
print('BB_cutoff old:', BB_cutoff)
print('AB_cutoff old:', AB_cutoff)