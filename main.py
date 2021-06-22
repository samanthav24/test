from display_all import displayRegression
from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from data_process import *
from init import *
from display_all import *
import os


#------------------------ INITIALIZATION ---------------------------

Linear_regression = True
Binary_classification = False
Binary_classification_single_feature = False

iterations = 100000
dump_interval = 1000

#------------------------ DATA PREPARATION --------------------------

features = []
timesteps = []

if Binary_classification or Binary_classification_single_feature:

    try:
        with open('./saves/Binary_features.npy', 'rb') as f:
            features, timesteps, rdfYoung, rdfOld = np.load(f, allow_pickle=True)
            print('All features loaded..')

    except:
        # iterate through all dump files    
        for file in os.listdir(os.fsencode(".\\dump\\young\\")):
            filename = os.fsdecode(file)
            if filename.endswith(".YOUNG"): 

                rdfNew, featuresNew = extract_features(".\\dump\\young\\", filename, 50000, 50)
                timestepsNew = np.zeros(len(featuresNew))

                features.extend(featuresNew)
                timesteps.extend(timestepsNew)

                try:
                    rdfYoung += rdfNew
                except:
                    rdfYoung = rdfNew

        # iterate through all dump files    
        for file in os.listdir(os.fsencode(".\\dump\\old\\")):
            filename = os.fsdecode(file)
            if filename.endswith(".OLD"): 

                rdfNew, featuresNew = extract_features(".\\dump\\old\\", filename, 1000000, 1000)
                timestepsNew = np.ones(len(featuresNew))

                features.extend(featuresNew)
                timesteps.extend(timestepsNew)

                try:
                    rdfOld += rdfNew
                except:
                    rdfOld = rdfNew

    rdfYoung /= 10
    rdfOld /= 10

    # save the features in a npy file
    with open('./saves/Binary_features.npy', 'wb') as f:
        np.save(f, [features, timesteps, rdfYoung, rdfOld])

    displayBinary(features, rdfYoung, rdfOld)


if Linear_regression:

    try:
        with open('.\saves\Regression_features.npy', 'rb') as f:
            features, timesteps, rdf = np.load(f, allow_pickle=True)
            print('All features loaded..')

    except:
        # iterate through all dump files    
        for file in os.listdir(os.fsencode(".\\dump\\full\\")):
            filename = os.fsdecode(file)
            if filename.endswith(".ATOM"): 

                rdfNew, featuresNew = extract_features(".\\dump\\full\\", filename, 5000000, 1000)
                timestepsNew = np.zeros(len(featuresNew))

                features.extend(featuresNew)
                timesteps.extend(timestepsNew)

                try:
                    rdf += rdfNew
                except:
                    rdf = rdfNew

        rdf /= 5

        # save the features in a npy file
        with open('./saves/Regression_features.npy', 'wb') as f:
            np.save(f, [features, timesteps, rdf])

    displayRegression(features, rdf)

#------------------------ PREDICTION ------------------------------

if Linear_regression:
    # first degree polynomial regression
    linear_regression(features, timesteps, params['test_ratio'], degree=1)
    # second degree polynomial regression
    linear_regression(features, timesteps, params['test_ratio'], degree=2)

if Binary_classification:
    logistic_regression(features, timesteps, params['test_ratio'])

if Binary_classification_single_feature:
    logistic_regression_single_feature(features, timesteps, params['test_ratio'])

