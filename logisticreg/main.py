# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 00:21:21 2015

@author: young
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def load_txt(filename):
    train_x = []
    train_y = []
    with open(filename) as f:
        alldata = f.readlines()
    for line in alldata:
        line_arr = line.strip().split()
        train_x.append([1.0,float(line_arr[0]),float(line_arr[1])])
        train_y.append(float(line_arr[2]))        
    return  train_x, train_y

def load_csv(filename):
    train_x = []
    train_y = []
    with open(filename) as f:
        alldata = f.readlines()
    for line in alldata:
        line_arr = line.strip().split()
        train_x.append([1.0,float(line_arr[0])])
        train_y.append(float(line_arr[1]))
    return  np.mat(train_x), np.mat(train_y).transpose() 
    
def sigmoid(z):
    return 1.0/(1+ sp.exp(-z))

def compute_cost(x,y,theta):
    h = x * theta
    sqErrors = h - y
    J = (1.0 / 2 ) * (sqErrors.transpose() * sqErrors)
    return J

def gradAscent(dataIn, classLabels):
    dataMatrix = np.mat(dataIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxInter = 500
    weights = np.ones((n,1))
    for k in range(maxInter):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * (dataMatrix).transpose() * error
    return weights
    
    

def gradient_descent(x,y,theta,alpha,num_iters):
    J_history = np.zeros(shape=(num_iters,1))
    for i in range(num_iters):
        h = x * theta
        print h
        sqErrors = h - y
        for j in range(theta.shape[0]):
            theta[j][0] = theta[j][0] - alpha*(sqErrors.transpose() * x[:,j]) 
        J_history[i, 0] = compute_cost(x, y, theta)    
    return theta, J_history
 
def plotBestFit(dataMat, labelMat,wei):
    weights = wei.getA()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker = 's')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()    
           
if __name__ == '__main__':
    train_x, train_y = load_txt('./data/testSet.txt')
    weights = gradAscent(train_x, train_y)
    plotBestFit(train_x, train_y, weights)
    


        
        

