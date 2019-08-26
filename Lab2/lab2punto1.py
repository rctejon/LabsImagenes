#Librerias utilizadas
import os, glob
import requests
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray
import skimage

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