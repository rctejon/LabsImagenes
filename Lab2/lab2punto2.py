import os
import requests
import numpy as np
import  nibabel as nib
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import glob

# Variables para el manejo de path
file_start='Brats17_2013_10_1_'
file_ends=['flair','t1','t1ce','t2','seg']
end=file_ends[0]

# Se carga un MRI (Flair) del paciente
mri=nib.load(os.path.join('data',file_start+end,file_start+end+'.nii'))

# Se declara el arreglo donde se guardaran los MRI del paciente
size=(240,240,155,4)
vol = np.zeros(size,dtype=np.single)

# Se imprime el tipo de dato de un MRI
print(mri.get_data_dtype())

# Se buscan todos los elementos de la carpeta data que empiezen por Brats17_2013_10_1_
i=0
for folder in list(glob.glob(os.path.join('data',file_start+'*'))):
    if('seg' in folder):
        continue
    # Se busca el archivo nii de una modalidad
    niiFile=list(glob.glob(os.path.join(folder,'*.nii')))
    print(folder,niiFile)
    # Se carga el nii
    mri=nib.load(os.path.join(niiFile[0]))
    # Se guarda en vol
    vol[:,:,:,i]=mri.get_fdata()
    i+=1

# Visualizacion de los cortes de cada modalidad de MRI
for corte in range(0,155):
    i1=plt.subplot(221)
    plt.imshow(vol[:,:,corte,0],cmap='gray')
    plt.tight_layout()
    i2=plt.subplot(222)
    plt.imshow(vol[:,:,corte,1],cmap='gray')
    plt.tight_layout()
    i3=plt.subplot(223)
    plt.imshow(vol[:,:,corte,2],cmap='gray')
    i4=plt.subplot(224)
    plt.imshow(vol[:,:,corte,3],cmap='gray')
    plt.tight_layout()
    i1.set_title('Flair')
    i2.set_title('T1')
    i3.set_title('T1CE')
    i4.set_title('T2')
    plt.draw()
    plt.pause(0.000001)
    plt.clf()