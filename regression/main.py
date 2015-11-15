# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:02:56 2015

@author: young
"""
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    dataArr = []; labelArr = []
    with open(filename) as f:
        all_data = f.readlines()
    for line in all_data:
        lineArr = []        
        curLine = line.strip().split('\t')
        for i in range(len(curLine) -1):    
            lineArr.append(float(curLine[i]))
        dataArr.append(lineArr)
        labelArr.append(float(curLine[-1]))
    return dataArr, labelArr
    
def stand_regression(x,y):
    xMat = np.mat(x)
    yMat = np.mat(y).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:        
        print "This matrix is singular, cannot do inverse"
        return 
    ws = xTx.I * (xMat.T * yMat)
    return ws
    
def plot_fit(x,y,w):
    xMat = np.mat(x)
    yMat = np.mat(y)
    fit_Mat = xMat * w
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1],yMat)
    plt.plot(xMat[:,1], fit_Mat)
    plt.show()
    
if __name__ == '__main__':
    train_data, train_label = load_data('./data/ex0.txt')
    ws = stand_regression(train_data, train_label)
    plot_fit(train_data, train_label,ws)
    