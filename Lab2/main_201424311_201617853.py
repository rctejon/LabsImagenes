#Librerias utilizadas
import os, glob
import requests
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray
import skimage
import scipy.io

print("Inicia punto 1")

#Descargar imagen a color de internet
image_url = "https://cdn.shopify.com/s/files/1/1280/3657/products/GM-BA-SET01_1_e5745bf6-253a-4a83-a8d1-5d55ee691e5d_1024x1024.jpg?v=1548113518"
r = requests.get(image_url)
with open("imagen.png", "wb") as f:
    f.write(r.content)
    
#Cargar la imagen en la variable image y convertirla a escala de grises
image = rgb2gray(io.imread(os.path.join("imagen.png")))

#Visualizacion de la imagen e histograma correspondiente
i = plt.subplot(3, 3, 1)
i.set_title("Imagen en escala de grises")
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
i = plt.subplot(3, 3, 2)
i.set_title("Histograma")
plt.hist(image.flatten())

#Calculo del umbral de binarizacion de acuerdo al metodo de Otsu
otsu_threshold = skimage.filters.threshold_otsu(image)
image_otsu = image.copy()
mask_otsu = image < otsu_threshold
mask_otsu2 = image > otsu_threshold
image_otsu[mask_otsu]=1
image_otsu[mask_otsu2]=0
i = plt.subplot(3, 3, 4)
i.set_title("Binarizacion con Otsu")
plt.imshow(image_otsu, cmap='gray', vmin=0, vmax=1)

#Segmentacion con umbral arbitrario de 0.3
random_threshold = 0.3
image_random = image.copy()
mask_random = image < random_threshold
mask_random2 = image > random_threshold
image_random[mask_random]=1
image_random[mask_random2]=0
i = plt.subplot(3, 3, 5)
i.set_title("Binarizacion con umbral 0.3")
plt.imshow(image_random, cmap='gray', vmin=0, vmax=1)

#Segmentacion con umbral determinado por el percentil 80 de las intensidades
percentile_threshold = np.percentile(image, 80)
image_percentile = image.copy()
mask_percentile = image < percentile_threshold
mask_percentile2 = image > percentile_threshold
image_percentile[mask_percentile]=1
image_percentile[mask_percentile2]=0
i = plt.subplot(3, 3, 6)
i.set_title("Binarizacion con percentil 80")
plt.imshow(image_percentile, cmap='gray', vmin=0, vmax=1)

#Reemplazar los pixeles de las monedas con el color promedio del fondo segun cada amscara hallada
result_otsu = image.copy()
result_otsu[mask_otsu] = 0.75
i = plt.subplot(3, 3, 7)
i.set_title("Reemplazo de pixels con Otsu")
plt.imshow(result_otsu, cmap='gray', vmin=0, vmax=1)

result_random = image.copy()
result_random[mask_random] = 0.75
i = plt.subplot(3, 3, 8)
i.set_title("Reemplazo de pixels con umbral 0.3")
plt.imshow(result_random, cmap='gray', vmin=0, vmax=1)

result_percentile = image.copy()
result_percentile[mask_percentile] = 0.75
i = plt.subplot(3, 3, 9)
i.set_title("Reemplazo de pixels con percentil 80")
plt.imshow(result_percentile, cmap='gray', vmin=0, vmax=1)
plt.show()

input("Press Enter to continue...")

print("Inicia punto 2")

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
    
input("Press Enter to continue...")
    
print("Inicia punto 3")

# Se lee el archivo .mat
mat = scipy.io.loadmat('jaccard.mat')

# Se guarda en variables la segmentación y la segmentación perfecta
real = mat['GroundTruth']
seg = mat['Segmentation']

intersection = 0
union = 0

# Se calculan los elementos en la intersección y en la unión
for i in range(len(seg)):
    for j in range(len(seg[i])):
        for k in range(len(seg[i,j])):
            if seg[i,j,k]==1 and real[i,j,k]==1:
                intersection+=1
            if seg[i,j,k]==1 or real[i,j,k]==1:
                union+=1

# Calculo indice de jaccard
print(f'Indice de jaccard {intersection/union}')