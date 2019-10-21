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
grad = morphology.dilation(image)-morphology.erosion(image)

plt.imshow(grad, cmap="gray")
plt.show()

minimos = [0,1,5,10,20,30,50,75,100]

conf = 331

for h in minimos:
    # plt.subplot(conf)
    # plt.title(h)
    markers=morphology.h_minima(grad,h)
    ws=morphology.watershed(grad,markers)
    plt.imshow(ws, cmap="gray")
    plt.show()
    conf+=1

plt.show()


    