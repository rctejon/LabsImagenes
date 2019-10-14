#Librerias utilizadas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob
import requests, zipfile
import skimage.io as io
import scipy.signal as sc
from sklearn.metrics import accuracy_score
from skimage.color import rgb2gray
import scipy.io as scio
import scipy
import scipy.ndimage as sc


#
# Punto 5.1
#
print('Inicia punto 5.1')


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

#
# Punto 5.2
#
print('Inicia punto 5.2')


#Se descarga la imagen del link y se guarda como beagle.jpg
ang = 'https://upload.wikimedia.org/wikipedia/commons/3/30/Cerebral_angiography%2C_arteria_vertebralis_sinister_injection.JPG'
r = requests.get(ang)
with open(os.path.join('ims','angiography.jpg'), "wb") as f:
    f.write(r.content)

#Se convierte la imagen a escala de grises
image = rgb2gray(io.imread(os.path.join('ims','angiography.jpg')))

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

#
# Punto 5.3
#
print('Inicia punto 5.3')

def MyAdaptMedian_201424311_201617853(image, window_size, max_window_size):
    #Calcular numero de columnas a agregar
    addColsNum=int((max_window_size-1)/2)
    #Se crea una copia de la imagen original y Se agrega un marco a la imagen para que la 𝑣𝑒𝑛𝑡𝑎𝑛𝑎max ajuste
    newImage=np.zeros((image.shape[0]+2*addColsNum,image.shape[1]+2*addColsNum)).astype(np.uint8)
    newImage[addColsNum:len(newImage)-addColsNum,addColsNum:len(newImage[0])-addColsNum]=image
   
    #Se crea arreglo con las medidas de la imagen donde iremos guardando la imagen a retornar
    returnImage=np.zeros((image.shape[0]+2*addColsNum,image.shape[1]+2*addColsNum)).astype(np.uint8)

    #Para cada uno de los píxeles
    i=addColsNum
    i2=addColsNum
    actualWindowSize = int((window_size-1)/2)
    while i < len(newImage)-addColsNum:
        #se elige un tamaño de ventana (empezando con una 𝑣𝑒𝑛𝑡𝑎𝑛𝑎min)
        #se hace la ventana alrededor de ese pixel (ahora pixel central con intensidad 𝑧𝑥𝑦)
        window=newImage[i-actualWindowSize:i+actualWindowSize+1,i2-actualWindowSize:i2+actualWindowSize+1]
        #se calcula el valor mínimo 𝑧min, valor máximo 𝑧m𝑎𝑥 y mediana 𝑧𝑚𝑒𝑑 de las intensidades de la ventana
        sortWindow=np.sort(window, axis=None)
        zmax = sortWindow[len(sortWindow)-1]
        zmin = sortWindow[0]
        zmed = sortWindow[int((len(sortWindow)+1)/2)]
        #Se calcula 𝐴1 restando 𝑧𝑚𝑖𝑛 a 𝑧𝑚𝑒𝑑 Se calcula 𝐴2 restando 𝑧𝑚𝑎𝑥 a 𝑧𝑚𝑒d
        A1 = zmed-zmin
        A2 = int(zmed)-int(zmax)
        if A1>0 and A2<0:
            #Se calcula 𝐵1 restando 𝑧𝑚𝑖𝑛 a 𝑧𝑥𝑦 y Se calcula 𝐵2 restando 𝑧𝑚𝑎𝑥 a 𝑧𝑥y
            B1 = int(newImage[i,i2])-int(zmin)
            B2 = int(newImage[i,i2])-int(zmax)
            if B1>0 and B2<0:
                returnImage[i,i2]=newImage[i,i2]
            else:
                returnImage[i,i2]=zmed
            #se pasa al siguiente pixel
            actualWindowSize=int((window_size-1)/2)
            if  i2<len(newImage[0])-addColsNum:
                i2+=1
            else:
                i2=addColsNum
                i+=1
        else:
            #se aumenta el tamaño de la ventana en 2
            actualWindowSize+=1
            if actualWindowSize>(max_window_size-1)/2:
                #actualiza 𝑍𝑥𝑦 en la copia de la imagen con 𝑍𝑚𝑒d
                returnImage[i,i2]=zmed
                #se pasa al siguiente pixel
                actualWindowSize=int((window_size-1)/2)
                if  i2<len(newImage[0])-addColsNum:
                    i2+=1
                else:
                    i2=addColsNum
                    i+=1
    #Se retorna el centro de la imagen
    return returnImage[addColsNum:len(newImage)-addColsNum,addColsNum:len(newImage[0])-addColsNum]

#Funcion para descargar archivos desde google drive
# def download_file_from_google_drive(id, destination):
#     URL = "https://docs.google.com/uc?export=download"

#     session = requests.Session()

#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = get_confirm_token(response)

#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)

#     save_response_content(response, destination)    

# def get_confirm_token(response):
#     for key, value in response.cookies.items():
#         if key.startswith('download_warning'):
#             return value

#     return None

# def save_response_content(response, destination):
#     CHUNK_SIZE = 32768

#     with open(destination, "wb") as f:
#         for chunk in response.iter_content(CHUNK_SIZE):
#             if chunk: # filter out keep-alive new chunks
#                 f.write(chunk)

# #Descarga del .zip desde google drive y posterior extraccion de las imagenes
# file_name = 'ims.zip'
# download_file_from_google_drive('1-_C3WZlXDq5Awf2OvVULkCpF_p14Wwv_', file_name)
# with zipfile.ZipFile(file_name, 'r') as z:
#     z.extractall()

im1 = io.imread(os.path.join('ims',"im1.jpg")).astype(np.uint8)
im2 = io.imread(os.path.join('ims',"im2.jpg")).astype(np.uint8)

plt.suptitle('Median Adaptative Filter - Window Size 15 - Max Window Size 15')
plt.subplot(221)
plt.title('Original Im1')
plt.imshow(im1, cmap='gray')
plt.subplot(222)
plt.title('Median Adaptative Filter Im1')
plt.imshow(MyAdaptMedian_201424311_201617853(im1,15,15), cmap='gray')
plt.subplot(223)
plt.title('Original Im2')
plt.imshow(im2, cmap='gray')
plt.subplot(224)
plt.title('Median Adaptative Filter Im2')
plt.imshow(MyAdaptMedian_201424311_201617853(im2,15,15), cmap='gray')
plt.show()

plt.suptitle('Median Adaptative Filter - Window Size 9 - Max Window Size 15')
plt.subplot(221)
plt.title('Original Im1')
plt.imshow(im1, cmap='gray')
plt.subplot(222)
plt.title('Median Adaptative Filter Im1')
plt.imshow(MyAdaptMedian_201424311_201617853(im1,9,15), cmap='gray')
plt.subplot(223)
plt.title('Original Im2')
plt.imshow(im2, cmap='gray')
plt.subplot(224)
plt.title('Median Adaptative Filter Im2')
plt.imshow(MyAdaptMedian_201424311_201617853(im2,9,15), cmap='gray')
plt.show()

plt.suptitle('Median Adaptative Filter - Window Size 5 - Max Window Size 15')
plt.subplot(221)
plt.title('Original Im1')
plt.imshow(im1, cmap='gray')
plt.subplot(222)
plt.title('Median Adaptative Filter Im1')
plt.imshow(MyAdaptMedian_201424311_201617853(im1,5,15), cmap='gray')
plt.subplot(223)
plt.title('Original Im2')
plt.imshow(im2, cmap='gray')
plt.subplot(224)
plt.title('Median Adaptative Filter Im2')
plt.imshow(MyAdaptMedian_201424311_201617853(im2,5,15), cmap='gray')
plt.show()

plt.suptitle('Median Adaptative Filter - Window Size 3 - Max Window Size 5 ')
plt.subplot(221)
plt.title('Original Im1')
plt.imshow(im1, cmap='gray')
plt.subplot(222)
plt.title('Median Adaptative Filter Im1')
plt.imshow(MyAdaptMedian_201424311_201617853(im1,3,5), cmap='gray')
plt.subplot(223)
plt.title('Original Im2')
plt.imshow(im2, cmap='gray')
plt.subplot(224)
plt.title('Median Adaptative Filter Im2')
plt.imshow(MyAdaptMedian_201424311_201617853(im2,3,5), cmap='gray')
plt.show()

plt.suptitle('Gaussian Filter - Sigma 1 ')
plt.subplot(221)
plt.title('Original Im1')
plt.imshow(im1, cmap='gray')
plt.subplot(222)
plt.title('Gaussian Filter Im1')
plt.imshow(sc.gaussian_filter(im1,sigma=1), cmap='gray')
plt.subplot(223)
plt.title('Original Im2')
plt.imshow(im2, cmap='gray')
plt.subplot(224)
plt.title('Gaussian Filter Im2')
plt.imshow(sc.gaussian_filter(im2,sigma=1), cmap='gray')
plt.show()

plt.suptitle('Gaussian Filter - Sigma 2 ')
plt.subplot(221)
plt.title('Original Im1')
plt.imshow(im1, cmap='gray')
plt.subplot(222)
plt.title('Gaussian Filter Im1')
plt.imshow(sc.gaussian_filter(im1,sigma=2), cmap='gray')
plt.subplot(223)
plt.title('Original Im2')
plt.imshow(im2, cmap='gray')
plt.subplot(224)
plt.title('Gaussian Filter Im2')
plt.imshow(sc.gaussian_filter(im2,sigma=2), cmap='gray')
plt.show()

plt.suptitle('Gaussian Filter - Sigma 3 ')
plt.subplot(221)
plt.title('Original Im1')
plt.imshow(im1, cmap='gray')
plt.subplot(222)
plt.title('Gaussian Filter Im1')
plt.imshow(sc.gaussian_filter(im1,sigma=3), cmap='gray')
plt.subplot(223)
plt.title('Original Im2')
plt.imshow(im2, cmap='gray')
plt.subplot(224)
plt.title('Gaussian Filter Im2')
plt.imshow(sc.gaussian_filter(im2,sigma=3), cmap='gray')
plt.show()

plt.suptitle('Best Filtes ')
plt.subplot(221)
plt.title('Original Im1')
plt.imshow(im1, cmap='gray')
plt.subplot(222)
plt.title('Median Adaptative Filter - Window Size 3 - Max Window Size 5')
plt.imshow(MyAdaptMedian_201424311_201617853(im1,3,5), cmap='gray')
plt.subplot(223)
plt.title('Original Im2')
plt.imshow(im2, cmap='gray')
plt.subplot(224)
plt.title('Gaussian Filter Sigma=2')
plt.imshow(sc.gaussian_filter(im2,sigma=2), cmap='gray')
plt.show()