#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests, zipfile
import skimage.io as io
import scipy.signal as sc
import pdb

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
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

#Descarga del .zip desde google drive y posterior extraccion de las imagenes
file_name = 'ims.zip'
download_file_from_google_drive('1-_C3WZlXDq5Awf2OvVULkCpF_p14Wwv_', file_name)
with zipfile.ZipFile(file_name, 'r') as z:
    z.extractall()

im1 = io.imread(os.path.join('ims',"im1.jpg")).astype(np.uint8)
im2 = io.imread(os.path.join('ims',"im2.jpg")).astype(np.uint8)

plt.suptitle('Window Size 7 - Im1')
plt.subplot(121)
plt.title('Original')
plt.imshow(im1, cmap='gray')
plt.subplot(122)
plt.title('Median Adaptative Filter')
plt.imshow(MyAdaptMedian_201424311_201617853(im1,3,15), cmap='gray')
plt.show()

plt.suptitle('Window Size 7 - Im2')
plt.subplot(121)
plt.title('Original')
plt.imshow(im2, cmap='gray')
plt.subplot(122)
plt.title('Median Adaptative Filter')
plt.imshow(MyAdaptMedian_201424311_201617853(im2,3,15), cmap='gray')
plt.show()

# plt.suptitle('Window Size 9 - Im1')
# plt.subplot(121)
# plt.title('Original')
# plt.imshow(im1, cmap='gray')
# plt.subplot(122)
# plt.title('Median Adaptative Filter')
# plt.imshow(MyAdaptMedian_201424311_201617853(im1,9,15), cmap='gray')
# plt.show()

# plt.suptitle('Window Size 9 - Im2')
# plt.subplot(121)
# plt.title('Original')
# plt.imshow(im2, cmap='gray')
# plt.subplot(122)
# plt.title('Median Adaptative Filter')
# plt.imshow(MyAdaptMedian_201424311_201617853(im2,9,15), cmap='gray')
# plt.show()

# plt.suptitle('Window Size 11 - Im1')
# plt.subplot(121)
# plt.title('Original')
# plt.imshow(im1, cmap='gray')
# plt.subplot(122)
# plt.title('Median Adaptative Filter')
# plt.imshow(MyAdaptMedian_201424311_201617853(im1,11,15), cmap='gray')
# plt.show()

# plt.suptitle('Window Size 11 - Im2')
# plt.subplot(121)
# plt.title('Original')
# plt.imshow(im2, cmap='gray')
# plt.subplot(122)
# plt.title('Median Adaptative Filter')
# plt.imshow(MyAdaptMedian_201424311_201617853(im2,11,15), cmap='gray')
# plt.show()

# plt.suptitle('Window Size 13 - Im1')
# plt.subplot(121)
# plt.title('Original')
# plt.imshow(im1, cmap='gray')
# plt.subplot(122)
# plt.title('Median Adaptative Filter')
# plt.imshow(MyAdaptMedian_201424311_201617853(im1,13,15), cmap='gray')
# plt.show()

# plt.suptitle('Window Size 13 - Im2')
# plt.subplot(121)
# plt.title('Original')
# plt.imshow(im2, cmap='gray')
# plt.subplot(122)
# plt.title('Median Adaptative Filter')
# plt.imshow(MyAdaptMedian_201424311_201617853(im2,13,15), cmap='gray')
# plt.show()