#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io
import scipy.signal as sc
import pdb
import scipy.io as scio

mat = scio.loadmat('challenge_results.mat')

def myACA_201424311_201617853(gt,pred, print_matrix=True):
    confMatrix = np.zeros((5,5))
    ACA = 0
    for i in range(len(pred)):
        g = gt[i]-1
        p = pred[i]-1
        confMatrix[g,p]+=1    
    for row in confMatrix:
        suma = np.sum(row)
        row *= 1/suma
    for i in range(len(confMatrix)):
        ACA+=confMatrix[i,i]/len(confMatrix)
    if print_matrix:
        print(confMatrix)
        print(ACA)
    return (confMatrix,ACA)


gt = mat['gt'][0].astype(np.int8)
method1 = mat['method1'][0].astype(np.int8)
method2 = mat['method2'][0].astype(np.int8)
method3 = mat['method3'][0].astype(np.int8)

myACA_201424311_201617853(gt,method1)

