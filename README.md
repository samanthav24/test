# Predicting the age of glass in active and passive matter using machine learning techniques

The outline of this repository is as follows:
- main.py <br>
calls all functions to print and visualize results

- dataread.py <br>
reads the data with following measured quantities: position, force, angular momentum and torque (in each direction) <br>
functions:

    * read_data: reads a .atom file and returns the timesteps and a dictionary where the keys are
the measures quantities and the values are the corresponding data of that feature

- util.py <br>
consists of various utilization functions used on the data:
    * calc_distance: calculates the distances between two particles taking the periodic boundary into consideration
    * msd: calculates the mean square displacement for each timestep
    * vsd: calculates the variance square displacement for each timestep
    * mean_nn: calculates the mean nearest neighbour distance and the mean amount of neighbours within the cutoff
    * variance_nn: calculates the variance of the nearest neighbour distance and the variance of the amount of neighbours within the cutoff
    * calc_mean: calculates the mean of the norm of the vector per timestep
    * calc_variance: calculates the variance of the norm of the vector per timestep
    * calc_rdf: calculates the radial distribution function in a single timestep
    * calc_rdf_peaks: calculates the (summed over all timesteps) radial distribution function peak for AA, AB and BB
    * calc_cutoff: calculates the cutoff parameter r based on the radial distribution function

- visualise.py <br>
functions:

    * visualise: plots each measured quantity through time


- ML.py <br>
consists of various machine learning function used on the data:

    * linear_regression: predicts the age of a system with linear, or given a degree > 1, polynomial regression with regularization and returns this prediction
    * logistic_regression: binary classification method which predicts whether a system is young or old with regularization