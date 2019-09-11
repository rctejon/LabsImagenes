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

conf=0.
puntosP=[]
puntosR=[]
while conf<=1:
    TP=0
    FN=0
    for i in range(0,len(anotaciones)):
        anot = anotaciones[i]
        pred = predicciones[i]
        maxX = max(anot[0],pred[0])
        maxY = max(anot[1],pred[1])
        minX = min(anot[0]+50,pred[0]+50)
        minY = min(anot[1]+50,pred[1]+50)
        area=0
        if maxX<minX and maxY<minY:
            area = (minX-maxX)*(minY-maxY)
        if conf<area/2500:
            TP+=1
        else:
            FN+=1
    prec = TP/(TP+FN)
    reca = TP/len(anotaciones)
    puntosP.append(prec)
    puntosR.append(reca)
    conf+=0.1
print(puntosP,puntosR)
plt.plot(puntosR,puntosP)
plt.ylabel("Precisión")
plt.xlabel("Cobertura")
plt.show()
