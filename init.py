"""
file in which all the constant parameters can be found
"""

params = {
    
    # simulation parameters
    'particles'   : 1000,          # amount of particles
    'dimensions'  : 2,
    'density'     : 1.2,           # density of the particle ensemble
    'dt'          : 0.0001,
    'Lx'          : 28.9,          # box dimensions
    'Ly'          : 28.9,
    'A'           : 28.9 * 28.9,   # box area

    'rmax'        : 5,
    'dr'          : 0.02,

    # cutoffs for Q6
    'r_cutoff_yp' : 1.43,          # r cutoff for a young and passive glass
    'r_cutoff_op' : 1.42,             # r cutoff for an old and passive glass
    'r_cutoff_ya' : 1.44,          # r cutoff for a young and active glass (cutoff young: (53, 86, 43))
    'r_cutoff_oa' : 1.43,          # r cutoff for an old and active glass (cutoff old: (53, 87, 43))

    # ML parameters
    'test_ratio'  : 0.2
}

titles = ['mnn_distance', 'vnn_distance', 'mean_force', 'variance_force',
                                     'mnn_amount', 'vnn_amount', 
                                     'mean_q4_re', 'variance_q4_re', 
                                     'mean_q4_im', 'variance_q4_im',
                                     'mean_q5_re', 'variance_q5_re', 
                                     'mean_q5_im', 'variance_q5_im',
                                     'mean_q6_re', 'variance_q6_re', 
                                     'mean_q6_im', 'variance_q6_im',
                                     'mean_q7_re', 'variance_q7_re', 
                                     'mean_q7_im', 'variance_q7_im',
                                     'mean_q8_re', 'variance_q8_re', 
                                     'mean_q8_im', 'variance_q8_im',
                                     'edgesA5', 'edgesA6', 'edgesA7', 
                                     'edgesB5', 'edgesB6', 'edgesB7',
                                     'poi0', 'poi1', 'poi2', 'poi3', 'poi4', 'poi5', 'poi6', 'poi7',
                                     'poi8', 'poi9', 'poi10', 'poi11', 'poi12', 'poi13', 'poi14']
