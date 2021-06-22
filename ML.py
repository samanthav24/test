import numpy as np
from numpy.core.defchararray import asarray
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import precision_score, recall_score
from sklearn import linear_model
from datetime import datetime
from copy import deepcopy
from display_all import *
from init import *
import itertools
from sklearn.metrics import mean_absolute_error


"""
Machine learning methods available:
    - Linear regression(X, y, test_ratio)
    - Logistic regression(X, y, test_ratio)
"""


def linear_regression(X, y, test_ratio, degree):

    # parameters = [['l1', 'l2'], [0.01, 0.01, 0.1, 1, 10, 100, 1000]]
    parameters = [['l1'], [0.1]]
    configurations = list(itertools.product(*parameters))

    random_state = 42
    tol = 0.1
    max_iter = 100000

    for configuration in configurations:

        penalty, alpha = configuration

        # get polynomial features
        # poly = PolynomialFeatures(degree)
        # X_poly = poly.fit_transform(X)

        # divide in training and test set and predict
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)

        # standardize features with z-score
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        if penalty == 'l1':
            clf = linear_model.Lasso(alpha=alpha, max_iter=max_iter, tol=tol).fit(X_train, y_train)
        else:
            clf = linear_model.Ridge(alpha=alpha, max_iter=max_iter, tol=tol).fit(X_train, y_train)

        pred = clf.predict(X_test).astype(int)

    f = open("./results/regression/results.txt", "a")
    
    f.write("-" * 80 + '\n\n')
    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + '\n')
    f.write('LINEAR REGRESSION' + '\n\n')
    f.write('Parameters: \n')
    f.write('Random state = ' + str(random_state) + '\n')
    f.write('Alpha = ' + str(alpha) + '\n')
    f.write('Test ratio = ' + str(test_ratio) + '\n')
    # f.write('Degree = ' + str(degree) + '\n')
    f.write('Regularization = ' + penalty + '\n')
    f.write('\n')
    f.write('Parameters:' + str(clf.get_params()) + '\n')
    f.write('Predicted age: ' + str(pred) + '\n')
    f.write('Real age: ' + str(y_test) + '\n')
    f.write('\n')
    f.write('Weights: \n \n')
    weights = zip(abs(clf.coef_), titles, clf.coef_)
    weights = sorted(weights, reverse=True)
    for _, title, weight in weights:
        f.write(title + ': ' + str(weight) + '\n')
    f.write('\n')

    R2_test = clf.score(X_test, y_test)
    R2_train = clf.score(X_train, y_train)

    f.write('R2 training set:' + str(R2_train) + '\n')
    f.write('R2 test set:' + str(R2_test) + '\n')
    f.write('Mean absolute error:' + str(mean_absolute_error(y_test, pred)) + '\n')


def logistic_regression(X, y, test_ratio):

    parameters = [['l1', 'l2'], [0.001, 0.01, 0.1, 1, 10, 100]]
    configurations = list(itertools.product(*parameters))

    random_state = 42
    tol = 0.0001
    solver = 'saga'
    max_iter = 10000

    # get polynomial features
    degree = 2
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    # divide in training and test set and predict
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_ratio, random_state=random_state)

    # standardize features with z-score
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    for configuration in configurations:

        penalty, C = configuration

        clf = LogisticRegression(max_iter=max_iter, tol=tol, C=C, solver=solver, penalty=penalty).fit(X_train, y_train)
        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)
        weights = clf.coef_

        accuracy_train= clf.score(X_train, y_train)    
        accuracy_test= clf.score(X_test, y_test)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)

        f = open("./results/binary/results.txt", "a")

        f.write("-" * 80 + '\n\n')
        f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + '\n')
        f.write('LOGISTIC REGRESSION' + '\n\n')
        f.write('Parameters: \n')
        f.write('Polynomial degree:' + str(degree) + '\n')
        f.write('Random state = ' + str(random_state) + '\n')
        f.write('Tolerance = ' + str(tol) + '\n')
        f.write('Solver = ' + str(solver) + '\n')
        f.write('C = ' + str(C) + '\n')
        f.write('Penalty = ' + str(penalty) + '\n')
        f.write('Test ratio = ' + str(test_ratio) + '\n')
        f.write('Max iter = ' + str(max_iter) + '\n')
        f.write('\n')
        f.write('Predictions:' + str(pred) + '\n')
        f.write('\n')
        f.write('with probabilities:' + str(prob) + '\n')
        f.write('\n')
        f.write('Weights: \n \n')
        weights = zip(abs(weights[0]), titles, weights[0])
        weights = sorted(weights, reverse=True)
        for _, title, weight in weights:
            f.write(title + ': ' + str(weight) + '\n')
        f.write('\n')
        f.write('accuracy training set:' + str(accuracy_train) + '\n')
        f.write('accuracy test set:' + str(accuracy_test) + '\n')
        f.write('Precision:' + str(precision) + '\n')
        f.write('Recall:' + str(recall) + '\n')

        f.close()

def logistic_regression_single_feature(X, y, test_ratio):


    random_state = 42
    tol = 0.0001
    solver = 'saga'
    max_iter = 10000
    penalty = 'l2'
    C = 10

    # divide in training and test set and predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)

    # standardize features with z-score
    scaler = StandardScaler().fit(X_train)
    X_train = np.asarray(scaler.transform(X_train))
    X_test = np.asarray(scaler.transform(X_test))


    f = open("./results/binary/results.txt", "a")
    f.write("-" * 80 + '\n\n')
    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + '\n')
    f.write('LOGISTIC REGRESSION WITH SINGLE FEATURES' + '\n\n')
    f.write('Parameters: \n')
    f.write('Random state = ' + str(random_state) + '\n')
    f.write('Tolerance = ' + str(tol) + '\n')
    f.write('Solver = ' + str(solver) + '\n')
    f.write('C = ' + str(C) + '\n')
    f.write('Penalty = ' + str(penalty) + '\n')
    f.write('Test ratio = ' + str(test_ratio) + '\n')
    f.write('Max iter = ' + str(max_iter) + '\n')
    f.write('\n')

    for i in range(len(X_train[0])):
        clf = LogisticRegression(max_iter=max_iter, tol=tol, C=C, solver=solver, penalty=penalty).fit(X_train[:, i].reshape(-1, 1), y_train)
        pred = clf.predict(X_test[:, i].reshape(-1, 1))
        prob = clf.predict_proba(X_test[:, i].reshape(-1, 1))
        weights = clf.coef_

        accuracy_train= round(clf.score(X_train[:, i].reshape(-1, 1), y_train),2)    
        accuracy_test= round(clf.score(X_test[:, i].reshape(-1, 1), y_test),2)  
        precision = round(precision_score(y_test, pred),2)  
        recall = round(recall_score(y_test, pred),2)  

        f.write("-" * 80 + '\n\n')
        # f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + '\n')
        f.write(titles[i] + ' weight: ' + str(round(weights[0][0],2)) + chr(92)+ chr(92) + '\n')
        f.write('\n')
        f.write('accuracy training set:' + str(accuracy_train) + chr(92)+ chr(92) + '\n')
        f.write('accuracy test set:' + str(accuracy_test)+ chr(92)+ chr(92) + '\n')
        f.write('Precision:' + str(precision)+ chr(92)+ chr(92) + '\n')
        f.write('Recall:' + str(recall)+ chr(92)+ chr(92) + '\n')

    f.close()

