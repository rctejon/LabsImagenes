#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io
import scipy.signal as sc
import pdb
from skimage.color import rgb2gray

def MyCCorrelation_201424311_201617853(image,kernel,boundary_condition):
    #Calcular numero de columnas a agregar
    addColsNum=int((len(kernel)-1)/2)
    #Se crea una copia de la imagen con mas columnas agregadas
    newImage=np.zeros((image.shape[0]+2*addColsNum,image.shape[1]+2*addColsNum)).astype(np.float64)
    #Se crea arreglo con las medidas de la imaen agregada
    returnImage=np.zeros((image.shape[0]+2*addColsNum,image.shape[1]+2*addColsNum)).astype(np.float64)
    #Se asigna los valores de la imagen al centro de la imagen para crear las columnas de mas
    newImage[addColsNum:len(newImage)-addColsNum,addColsNum:len(newImage[0])-addColsNum]=image
    #En caso de recibir condiciones de frontera valid se deja la copia con los valores de la imagen
    if boundary_condition == 'valid':
        newImage = image
        returnImage= np.zeros_like(image)
    #En caso de recibir condiciones de frontera wrap se copian los valores de un lado al lado contrario de la imagen
    elif boundary_condition =='wrap':
        newImage[:addColsNum,:]=newImage[-2*addColsNum:-addColsNum,:]
        newImage[-addColsNum:,:]=newImage[addColsNum:2*addColsNum,:]
        newImage[:,:addColsNum]=newImage[:,-2*addColsNum:-addColsNum]
        newImage[:,-addColsNum:]=newImage[:,addColsNum:2*addColsNum]
    
    #Se realiza la cross correlacion
    for i in range(addColsNum,len(newImage)-addColsNum):
        for i2 in range(addColsNum,len(newImage[0])-addColsNum):
            cuadrado=newImage[i-addColsNum:i+addColsNum+1,i2-addColsNum:i2+addColsNum+1]
            mult=cuadrado*kernel
            returnImage[i,i2]=np.sum(mult)
    #Se retorna el centro de la imagen
    return returnImage[addColsNum:len(newImage)-addColsNum,addColsNum:len(newImage[0])-addColsNum]

#Se descarga y convierte a rgb una imagen de internet
husky = 'https://img.milanuncios.com/fg/2659/27/265927918_1.jpg?VersionId=RmbFrmcRlrZ6TOb_fQ23oeGThoI_rfAs'
r = requests.get(husky)
with open("husky.jpg", "wb") as f:
    f.write(r.content)
husky = rgb2gray(io.imread(os.path.join("husky.jpg")))

#Se inicializan los kernels
kernelA = np.array([1,1,1,0,0,0,0,0,0]).reshape(3,3)
kernelB = np.array([1,0,-1,2,0,-2,1,0,-1]).reshape(3,3)
kernelC = (1/9)*np.array([1,1,1,1,1,1,1,1,1]).reshape(3,3)

#Se ejecutan las funciones de croscorrelacion con el kernel A y condicion valid
image1=MyCCorrelation_201424311_201617853(husky,kernelA,'valid')
image2 = sc.correlate2d(husky,kernelA,mode='valid',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel A - valid')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()

#Se ejecutan las funciones de croscorrelacion con el kernel A y condicion wrap
image1=MyCCorrelation_201424311_201617853(husky,kernelA,'wrap')
image2 = sc.correlate2d(husky,kernelA,mode='same',boundary='wrap')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel A - wrap')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()

#Se ejecutan las funciones de croscorrelacion con el kernel A y condicion fill
image1=MyCCorrelation_201424311_201617853(husky,kernelA,'fill')
image2 = sc.correlate2d(husky,kernelA,mode='same',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel A - fill')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()


#Se ejecutan las funciones de croscorrelacion con el kernel B y condicion valid
image1=MyCCorrelation_201424311_201617853(husky,kernelB,'valid')
image2 = sc.correlate2d(husky,kernelB,mode='valid',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel B - valid')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()

#Se ejecutan las funciones de croscorrelacion con el kernel B y condicion wrap
image1=MyCCorrelation_201424311_201617853(husky,kernelB,'wrap')
image2 = sc.correlate2d(husky,kernelB,mode='same',boundary='wrap')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel B - wrap')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()

#Se ejecutan las funciones de croscorrelacion con el kernel B y condicion fill
image1=MyCCorrelation_201424311_201617853(husky,kernelB,'fill')
image2 = sc.correlate2d(husky,kernelB,mode='same',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel B - fill')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()

#Se ejecutan las funciones de croscorrelacion con el kernel C y condicion valid
image1=MyCCorrelation_201424311_201617853(husky,kernelC,'valid')
image2 = sc.correlate2d(husky,kernelC,mode='valid',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel C - valid')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()

#Se ejecutan las funciones de croscorrelacion con el kernel C y condicion wrap
image1=MyCCorrelation_201424311_201617853(husky,kernelC,'wrap')
image2 = sc.correlate2d(husky,kernelC,mode='same',boundary='wrap')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel C - wrap')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()

#Se ejecutan las funciones de croscorrelacion con el kernel C y condicion fill
image1=MyCCorrelation_201424311_201617853(husky,kernelC,'fill')
image2 = sc.correlate2d(husky,kernelC,mode='same',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel C - fill')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()