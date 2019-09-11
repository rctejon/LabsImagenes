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
    addColsNum=int((len(kernel)-1)/2)
    newImage=np.zeros((image.shape[0]+2*addColsNum,image.shape[1]+2*addColsNum)).astype(np.float64)
    returnImage=np.zeros((image.shape[0]+2*addColsNum,image.shape[1]+2*addColsNum)).astype(np.float64)
    newImage[addColsNum:len(newImage)-addColsNum,addColsNum:len(newImage[0])-addColsNum]=image
    if boundary_condition == 'valid':
        newImage = image
        returnImage= np.zeros_like(image)
    elif boundary_condition =='wrap':
        newImage[:addColsNum,:]=newImage[-2*addColsNum:-addColsNum,:]
        newImage[-addColsNum:,:]=newImage[addColsNum:2*addColsNum,:]
        newImage[:,:addColsNum]=newImage[:,-2*addColsNum:-addColsNum]
        newImage[:,-addColsNum:]=newImage[:,addColsNum:2*addColsNum]
    
    for i in range(addColsNum,len(newImage)-addColsNum):
        for i2 in range(addColsNum,len(newImage[0])-addColsNum):
            cuadrado=newImage[i-addColsNum:i+addColsNum+1,i2-addColsNum:i2+addColsNum+1]
            mult=cuadrado*kernel
            returnImage[i,i2]=np.sum(mult)
            #pdb.set_trace()
    return returnImage[addColsNum:len(newImage)-addColsNum,addColsNum:len(newImage[0])-addColsNum]


husky = 'https://img.milanuncios.com/fg/2659/27/265927918_1.jpg?VersionId=RmbFrmcRlrZ6TOb_fQ23oeGThoI_rfAs'
r = requests.get(husky)
with open("husky.jpg", "wb") as f:
    f.write(r.content)
husky = rgb2gray(io.imread(os.path.join("husky.jpg")))
husky2 = rgb2gray(io.imread(os.path.join("husky.jpg")))
#husky = np.array([0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.,0.1,0.2,0.3,0.4]).reshape(5,5)
# array = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]*20
# husky = (np.array(array)*(1/10)).reshape(20,20)

kernelA = np.array([1,1,1,0,0,0,0,0,0]).reshape(3,3)
kernelB = np.array([1,0,-1,2,0,-2,1,0,-1]).reshape(3,3)
kernelC = (1/9)*np.array([1,1,1,1,1,1,1,1,1]).reshape(3,3)
husky = np.random.rand(10, 10)
husky2 = husky.copy()
image1=MyCCorrelation_201424311_201617853(husky,kernelA,'valid')
image2 = sc.convolve2d(husky2,kernelA,mode='valid',boundary='fill')

print(husky.shape,image1.shape, image2.shape)
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.subplot(131)
plt.imshow(husky,cmap='gray')
plt.subplot(132)
plt.imshow(image1,cmap='gray')
plt.subplot(133)
plt.imshow(image2,cmap='gray')
plt.show()