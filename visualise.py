import matplotlib.pyplot as plt


def visualise(timesteps, **kwargs): 
    """
    visualises the data found
    """

    for key, value in kwargs.items():

        plt.scatter(timesteps, value)
        plt.xlabel('timesteps')
        plt.title(key)
        plt.show()