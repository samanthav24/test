from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from data_process import *
from init import *
import os
 
 
#------------------------ INITIALIZATION ---------------------------
 
iterations = 100000
dump_interval = 1000
full = False
 
#------------------------ DATA PREPARATION --------------------------
 
if not full:
    total = 0
    # iterate through all dump files    
    for file in os.listdir(os.fsencode(".\\dump\\young\\")):
        filename = os.fsdecode(file)
        if filename.endswith(".YOUNG"):
 
            total += 1 
 
            timesteps1, types, q6_re, q6_im, vor_area, vor_amn, Data = read_data(".\\dump\\young\\" + filename, 50000, 50)
            
            q6_re = calc_mean(q6_re)
            q6_im = calc_mean(q6_im)
 
            try:
                q6_re_yo += q6_re
                q6_im_yo += q6_im
            except:
                q6_re_yo = q6_re
                q6_im_yo = q6_im
 
    # iterate through all dump files    
    for file in os.listdir(os.fsencode(".\\dump\\old\\")):
        filename = os.fsdecode(file)
        if filename.endswith(".OLD"): 
 
            timesteps2, types, q6_re, q6_im, vor_area, vor_amn, Data = read_data(".\\dump\\old\\" + filename, 1000000, 1000)
            
            q6_re = calc_mean(q6_re)
            q6_im = calc_mean(q6_im)
 
            try:
                q6_re_ol += q6_re
                q6_im_ol += q6_im
            except:
                q6_re_ol = q6_re
                q6_im_ol = q6_im
 
    q6_re_yo /= total
    q6_im_yo /= total
    q6_re_ol /= total
    q6_im_ol /= total
 
    timesteps1 = [50*t for t in timesteps1]
    timesteps2 = [4000000 + t*1000 for t in timesteps2]
 
    ax1 = plt.subplot(121)
    plt.title('mean q6 real young')
    ax2 = plt.subplot(122)
    plt.title('mean q6 real old')
 
    ax1.plot(timesteps1, q6_re_yo)
    ax2.plot(timesteps2, q6_re_ol)
 
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax1.set_yticklabels([])
 
    plt.xlabel('t')
    plt.ylabel('q6')
    plt.show()
    
    ax1 = plt.subplot(121)
    plt.title('mean q6 imaginary young')
    ax2 = plt.subplot(122)
    plt.title('mean q6 imaginary old')
        
    ax1.plot(timesteps1, q6_im_yo)
    ax2.plot(timesteps2, q6_im_ol)
 
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax1.set_yticklabels([])
    
    plt.xlabel('t')
    plt.ylabel('q6')
 
    plt.show()
 
 
 
 
 
 
if full:
    # iterate through all dump files    
    for file in os.listdir(os.fsencode(".\\dump\\full\\")):
        filename = os.fsdecode(file)
        if filename.endswith(".ATOM"): 
 
            featuresNew = extract_features(".\\dump\\full\\", filename, 5000000, 1000)
            timestepsNew = np.zeros(len(featuresNew))
 
            features.append(featuresNew)
            timesteps.extend(timestepsNew)
 
