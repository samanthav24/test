import os
from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from init import *


iterations = 50000
dump_interval = 50

rdf = True
q6 = False
directory = ".\\dump\\young\\"
extension = ".YOUNG"

amount = 0

# iterate through all dump files    
for file in os.listdir(os.fsencode(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(extension): 

        savename = './/saves//' + filename.split(".")[0] + '-rdf.npy'

        # retrieve and process data
        timesteps, types, q6_re, q6_im, Data = read_data(directory + filename, iterations, dump_interval)
        Data = {'position': Data[0, :1], 'force': Data[1], 'q6_re': q6_re, 'q6_im': q6_im}
        types = types[:1]

        # declare variable for the first run
        if amount == 0:
            
            if rdf:
                gr = save_load(lambda: calc_avg_rdf(Data['position'], types), savename)

            if q6:
                mean_q6 = np.asarray((calc_mean(Data['q6_re']), calc_mean(Data['q6_im'])))
                var_q6 = np.asarray((calc_variance(Data['q6_re'], mean_q6[0]), calc_variance(Data['q6_im'], mean_q6[1])))

        else:
            if rdf:
                gr += save_load(lambda: calc_avg_rdf(Data['position'], types), savename)

            if q6:
                mean_q6 += np.asarray((calc_mean(Data['q6_re']), calc_mean(Data['q6_im'])))
                var_q6 += np.asarray((calc_variance(Data['q6_re'], mean_q6[0]), calc_variance(Data['q6_im'], mean_q6[1])))
        
        amount += 1

# visualise the found averages
if rdf:

    gr /= amount
    r = np.arange(0, params['rmax'] + params['dr'], params['dr'])
    plot_rdf(r, gr[0], 'grAA')
    plot_rdf(r, gr[1], 'grBB')
    plot_rdf(r, gr[2], 'grAB')

if q6:
    mean_q6 /= amount
    var_q6 /= amount
    visualise(timesteps, False, q6_mean_real=mean_q6[0], q6_mean_imaginary=mean_q6[1],
                         q6_variance_real=var_q6[0], q6_variance_imaginary=var_q6[1])
