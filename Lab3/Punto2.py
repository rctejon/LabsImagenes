#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io
import scipy.io as scio

mat = scio.loadmat('detection_results.mat')

anotaciones = mat['annotations']
predicciones = mat['predictions']
confianzas = mat['scores']
print(len(predicciones))
print(len(anotaciones))

conf=0.
puntosP=[]
puntosR=[]
while conf<=1:
    TP=0
    FP=0
    FN=0
    for i in range(0,len(anotaciones)):
        anot = anotaciones[i]
        pred = predicciones[i]
        score = confianzas[i][0]
        maxX = max(anot[0],pred[0])
        maxY = max(anot[1],pred[1])
        minX = min(anot[0]+50,pred[0]+50)
        minY = min(anot[1]+50,pred[1]+50)
        area=0
        if maxX<minX and maxY<minY:
            area = (minX-maxX)*(minY-maxY)
        if conf<=score and conf<area/2500:
            TP+=1
        elif conf>score and conf<area/2500:
            FP+=1
        elif conf<=score and conf>=area/2500 :
            FN+=1
    prec = TP/(TP+FP)
    reca = TP/(TP+FN)
    puntosP.append(prec)
    puntosR.append(reca)
    conf+=0.01
print(puntosP,puntosR)
plt.plot(puntosR,puntosP)
plt.show()
