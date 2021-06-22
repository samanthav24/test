import numpy as np
import matplotlib.pyplot as plt
from init import *

def displayBinary(features, rdfYoung, rdfOld):
    """
    saves all the features generated in main.py
    called from main.py - takes the average
    """
    print('test 1')

    # split the features in 20 parts - for each file
    features = np.array_split(features, 20)
    features = np.array_split(features, 2)

    # split into young and old and take the mean
    young_features = np.mean(features[0], axis=0)
    old_features = np.mean(features[1], axis=0)

    timestepsYoung = range(0, 50050, 50)
    timestepsOld = range(4000000, 5001000, 1000)
    
    for i in range(len(young_features[0])):

        # clear any lingering plot
        plt.clf()

        ax1 = plt.subplot(121)
        plt.title(titles[i] + ' young')
        ax2 = plt.subplot(122)
        plt.title(titles[i] + ' old')

        ax1.plot(timestepsYoung, young_features[:, i])
        ax2.plot(timestepsOld, old_features[:, i])

        ax1.get_shared_y_axes().join(ax1, ax2)
        ax1.set_yticklabels([])
        
        plt.xlabel('t')

        plt.savefig('./results/binary/' + titles[i] + '.png')

    plt.clf()
    print('test 2')

    r = np.arange(0,params['rmax'] + params['dr'], params['dr'])

    plt.plot(r, rdfYoung[0], label='t=0')
    plt.plot(r, rdfOld[0], label='t=4000000')
    plt.title('grAA with poi')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.savefig('./results/binary/grAA.png')
    plt.clf()

    plt.plot(r, rdfYoung[1], label='t=0')
    plt.plot(r, rdfOld[1], label='t=4000000')
    plt.title('grBB with poi')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.legend()
    plt.savefig('./results/binary/grBB.png')
    plt.clf()

    plt.plot(r, rdfYoung[2], label='t=0')
    plt.plot(r, rdfOld[2], label='t=4000000')
    plt.title('grAB with poi')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.legend()
    plt.savefig('./results/binary/grAB.png')
    plt.clf()

    print('test 3')
def displayRegression(features, rdf):
    """
    saves all the features generated in main.py
    called from main.py - takes the average
    """

    # split the features in 5 parts - for each file
    features = np.array_split(features, 5)

    # take the mean
    features = np.mean(features, axis=0)

    timesteps = range(0, 5001000, 1000)
    
    for i in range(len(features[0])):

        # clear any lingering plot
        plt.clf()

        plt.plot(timesteps, features[:, i])
        plt.title(titles[i])
        
        plt.xlabel('t')

        plt.savefig('./results/regression/' + titles[i] + '.png')

    plt.clf()

    r = np.arange(0,params['rmax'] + params['dr'], params['dr'])

    plt.plot(r, rdf[0], label='t=0')
    plt.title('grAA with poi')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.savefig('./results/regression/grAA.png')
    plt.clf()

    plt.plot(r, rdf[1], label='t=0')
    plt.title('grBB with poi')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.legend()
    plt.savefig('./results/regression/grBB.png')
    plt.clf()

    plt.plot(r, rdf[2], label='t=0')
    plt.title('grAB with poi')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.legend()
    plt.savefig('./results/regression/grAB.png')
    plt.clf()