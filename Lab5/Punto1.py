# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:43:53 2019

@author: Mariana
"""

#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io
import scipy.signal as sc
import pdb
from skimage.color import rgb2gray

def pad(array, reference, offset):
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    result[offset:array.shape[0]+offset,offset:array.shape[1]+offset] = array
    return result

def MyGaussian_201424311_201617853(image, filter_size, sigma):
    #Calcular el rango del filtro [-n, n]
    n = (filter_size - 1)//2 
    
    #Calcular numero de columnas a agregar
    addColsNum = int((filter_size-1)/2)
    #Se crea una copia de la imagen original y Se agrega un marco a la imagen para que lajuste
    newImage = pad(image, np.zeros((image.shape[0]+2*addColsNum,image.shape[1]+2*addColsNum)).astype(np.uint8), addColsNum)
    #Se crea arreglo con las medidas de la imagen donde iremos guardando la imagen a retornar
    returnImage = np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
    
    #Se crea el grid que se va a usar para calcular el filtro
    s = np.arange(-n, n+1, 1)
    x, y = np.meshgrid(s, s)
    z = np.exp(-(x**2+y**2)/(2*sigma**2)) / (2*np.pi*sigma**2)
    
    #Se normalizan los coeficientes de la mascara para evitar modificar el nivel de gris
    mask = z / np.sum(z)

    #Para cada uno de los píxeles se aplica el filtro
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            window = newImage[addColsNum+i-n:addColsNum+i+n+1,addColsNum+j-n:addColsNum+j+n+1]
            returnImage[i][j] = np.sum(np.multiply(window, mask))
        
    #Se retorna el centro de la imagen
    return returnImage

#Se descarga la imagen del link y se guarda como beagle.jpg
beagle = 'https://www.petdarling.com/articulos/wp-content/uploads/2014/08/cachorro-beagle.jpg'
r = requests.get(beagle)
with open(os.path.join('ims','beagle.jpg'), "wb") as f:
    f.write(r.content)

#Se convierte la imagen a escala de grises
beagle = (256*rgb2gray(io.imread(os.path.join('ims','beagle.jpg')).astype(np.uint8))).astype(np.uint8)
   
#Parametros del filtro 
sigma = 1.2
filter_size = 7

plt.suptitle(f'Sigma = {sigma} Size = {filter_size}')
plt.subplot(121)
plt.title('Original')
plt.imshow(beagle, cmap='gray')
plt.subplot(122)
plt.title('GaussianFilter')
plt.imshow(MyGaussian_201424311_201617853(beagle,filter_size,sigma), cmap='gray')
plt.show()

#Parametros del filtro 
sigma = 5

plt.suptitle(f'Sigma = {sigma} Size = {filter_size}')
plt.subplot(121)
plt.title('Original')
plt.imshow(beagle, cmap='gray')
plt.subplot(122)
plt.title('GaussianFilter')
plt.imshow(MyGaussian_201424311_201617853(beagle,filter_size,sigma), cmap='gray')
plt.show()

#Parametros del filtro 
sigma = 10

plt.suptitle(f'Sigma = {sigma} Size = {filter_size}')
plt.subplot(121)
plt.title('Original')
plt.imshow(beagle, cmap='gray')
plt.subplot(122)
plt.title('GaussianFilter')
plt.imshow(MyGaussian_201424311_201617853(beagle,filter_size,sigma), cmap='gray')
plt.show()