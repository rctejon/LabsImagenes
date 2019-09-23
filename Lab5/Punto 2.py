#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io
import scipy.signal as sc
import scipy
import pdb
from skimage.color import rgb2gray

#Se descarga la imagen del link y se guarda como beagle.jpg
ang = 'https://upload.wikimedia.org/wikipedia/commons/3/30/Cerebral_angiography%2C_arteria_vertebralis_sinister_injection.JPG'
r = requests.get(ang)
with open(os.path.join('ims','angiography.jpg'), "wb") as f:
    f.write(r.content)

#Se convierte la imagen a escala de grises
image = rgb2gray(io.imread(os.path.join('ims','angiography.jpg')))

#Funcion para agregar padding de ceros a una imagen
def pad(array, reference, offset):
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    result[offset:array.shape[0]+offset,offset:array.shape[1]+offset] = array
    return result

#Creacion de los kernels
kernelA = np.array([0.0030,0.0133,0.0219,0.0133,0.0030,
                    0.0133,0.0596,0.0983,0.0596,0.0133,
                    0.0219,0.09843,0.1621,0.0983,0.0219,
                    0.0133,0.0596,0.0983,0.0596,0.0133,
                    0.0030,0.0133,0.0219,0.0133,0.0030]).reshape(5,5)
kernelB = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape(3,3)
kernelC = np.array([1,2,1,0,0,0,-1,-2,-1]).reshape(3,3)

R1 = scipy.ndimage.correlate(image, kernelB, mode='constant', cval=0.0)
R2 = scipy.ndimage.correlate(image, kernelC, mode='constant', cval=0.0)
R3 = np.sqrt(R1**2 + R2**2)

filterA = scipy.ndimage.correlate(image, kernelA, mode='constant', cval=0.0)
R4 = scipy.ndimage.correlate(filterA, kernelB, mode='constant', cval=0.0)
R5 = scipy.ndimage.correlate(filterA, kernelC, mode='constant', cval=0.0)
R6 = np.sqrt(R4**2 + R5**2)

for i in range(1,7):
    plt.subplot(230 + i)
    plt.title(f'R{i}')
    plt.imshow(eval(f'R{i}'), cmap='gray')
plt.tight_layout() 
plt.show()