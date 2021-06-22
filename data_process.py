from dataread import read_data
from util import *

def extract_features(directory, filename, iterations, dump_interval):
    
    timesteps, types, q4, q5, q6, q7, q8, Data = read_data(directory + filename, iterations, dump_interval)
    Data = {'position': Data[0], 'force': Data[1], 'q4': q4, 'q5': q5, 'q6': q6, 'q7': q7, 'q8': q8}

    savename = './/saves//' + filename.split(".")[0]

    # get the mean square displacement and the variance square displacement of the position data (deprecated)
    # msd = msd(Data['position'])
    # vsd = vsd(Data['position'], msd)

    # calculate the mean and variance nearest neighbour distance per timestep
    mnn_distance, mnn_amount = save_load(lambda: mean_nn(Data['position'], 1.43), savename + '-mean_nn.npy')
    vnn_distance, vnn_amount  = save_load(lambda: variance_nn(Data['position'], mnn_distance, mnn_amount, 1), savename + '-var_nn.npy')

    # calculate the mean and variance of the norm of the force
    mean_force = calc_mean(Data['force'])
    variance_force = calc_variance(Data['force'], mean_force)


    # calculate the mean and variance of the real q4 parameters
    mean_q4_re = calc_mean2(Data['q4'][:, :, 0])
    variance_q4_re = calc_variance2(Data['q4'][:, :, 0], mean_q4_re)

    # calculate the mean and variance of the imaginary q4 parameters
    mean_q4_im = calc_mean2(Data['q4'][:, :, 1])
    variance_q4_im = calc_variance2(Data['q4'][:, :, 1], mean_q4_im)

    # calculate the mean and variance of the real q5 parameters
    mean_q5_re = calc_mean2(Data['q5'][:, :, 0])
    variance_q5_re = calc_variance2(Data['q5'][:, :, 0], mean_q5_re)

    # calculate the mean and variance of the imaginary q5 parameters
    mean_q5_im = calc_mean2(Data['q5'][:, :, 1])
    variance_q5_im = calc_variance2(Data['q5'][:, :, 1], mean_q5_im)

    # calculate the mean and variance of the real q6 parameters
    mean_q6_re = calc_mean2(Data['q6'][:, :, 0])
    variance_q6_re = calc_variance2(Data['q6'][:, :, 0], mean_q6_re)

    # calculate the mean and variance of the imaginary q6 parameters
    mean_q6_im = calc_mean2(Data['q6'][:, :, 1])
    variance_q6_im = calc_variance2(Data['q6'][:, :, 1], mean_q6_im)

    # calculate the mean and variance of the real q7 parameters
    mean_q7_re = calc_mean2(Data['q7'][:, :, 0])
    variance_q7_re = calc_variance2(Data['q7'][:, :, 0], mean_q7_re)

    # calculate the mean and variance of the imaginary q7 parameters
    mean_q7_im = calc_mean2(Data['q7'][:, :, 1])
    variance_q7_im = calc_variance2(Data['q7'][:, :, 1], mean_q7_im)

    # calculate the mean and variance of the real q8 parameters
    mean_q8_re = calc_mean2(Data['q8'][:, :, 0])
    variance_q8_re = calc_variance2(Data['q8'][:, :, 0], mean_q8_re)

    # calculate the mean and variance of the imaginary q8 parameters
    mean_q8_im = calc_mean2(Data['q8'][:, :, 1])
    variance_q8_im = calc_variance2(Data['q8'][:, :, 1], mean_q8_im)

    # calculate the radial distribution function
    rdf = np.asarray(save_load(lambda: calc_rdf_all(Data['position'], types), savename + '-rdf.npy'))

    # calculate the points of interest of the radial distribution function
    poi = rdf_poi(rdf)

    rdf = rdf[:, 0]

    # calculate the count of n voronoi edges
    edgesA5, edgesA6, edgesA7, edgesB5, edgesB6, edgesB7 = calc_voronoi(timesteps, types, Data['position'])

    # prepare features in single array
    features = rdf, np.column_stack([mnn_distance, vnn_distance, mean_force, variance_force,
                                     mnn_amount, vnn_amount, 
                                     mean_q4_re, variance_q4_re, 
                                     mean_q4_im, variance_q4_im,
                                     mean_q5_re, variance_q5_re, 
                                     mean_q5_im, variance_q5_im,
                                     mean_q6_re, variance_q6_re, 
                                     mean_q6_im, variance_q6_im,
                                     mean_q7_re, variance_q7_re, 
                                     mean_q7_im, variance_q7_im,
                                     mean_q8_re, variance_q8_re, 
                                     mean_q8_im, variance_q8_im,
                                     edgesA5, edgesA6, edgesA7, 
                                     edgesB5, edgesB6, edgesB7,
                                     *poi])

    return features
