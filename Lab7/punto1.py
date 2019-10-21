#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io
from skimage.color import rgb2gray
from skimage import filters
from skimage import measure
from skimage import morphology

image_url = "https://www.thesprucepets.com/thmb/wNsACo_CfCKLw5zcgUOH9NCgqs8=/960x0/filters:no_upscale():max_bytes(150000):strip_icc()/fennec-fox-85120553-57ffe0d85f9b5805c2b03554.jpg"
r = requests.get(image_url)
with open("imagen.png", "wb") as f:
    f.write(r.content)
#Cargar la imagen en la variable image y mostrarla
image = rgb2gray(io.imread(os.path.join("imagen.png")))
val = filters.threshold_otsu(image)
image=image > val
def extractComponent(binary_image, labeled_image,i,j,k,conn):
    X0 = []
    X1 = np.full_like(labeled_image,False)
    X1[i][j]=True
    B = np.ones((3,3))
    if conn==4:
        B[0][0]=0
        B[0][2]=0
        B[2][0]=0
        B[2][2]=0
    while(not np.array_equal(X0,X1)):
        X0=np.copy(X1)
        X1= np.logical_and(morphology.binary_dilation(X0,B),binary_image)
    labeled_image= X1*k +np.multiply(np.invert(X1),labeled_image)
    return (labeled_image,X1)
def MyConnComp_201424311_201617853(binary_image, conn):
    labeled_image = np.zeros_like(binary_image)
    copy = np.copy(binary_image)
    pixel_labels =[]
    k=1
    for i in range(len(copy)):
        for j in range(len(copy[i])):
            if copy[i,j]:
                labeled_image, component =extractComponent(binary_image, labeled_image,i,j,k,conn)
                copy = np.logical_and(copy,np.invert(component))
                index_vector = []
                for i1 in range(len(component)):
                    for j1 in range(len(component[i1])):
                        if component[i1][j1]:
                            index_vector.append(i1*(len(component)-1)+j1)
                pixel_labels.append(index_vector)
                k+=1

    print(np.max(labeled_image))
    labeled_image = measure.label(binary_image,connectivity=(1 if conn==4 else 2 ))
    print(np.max(labeled_image))
    plt.imshow(labeled_image, cmap='gray')
    plt.show()
    return (labeled_image,pixel_labels)

MyConnComp_201424311_201617853(image,8)

plt.imshow( image, cmap='gray', interpolation='nearest')
plt.show()