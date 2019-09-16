#Librerias utilizadas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests, zipfile
import skimage.io as io
import scipy.signal as sc
from sklearn.metrics import accuracy_score
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

#Se inicializan los dos kernels a usar
kernelCuadrado = np.array([0]*25).reshape(5,5)
kernelCuadrado[1:4,1:4] = np.array([1]*9).reshape(3,3)
kernelCuadrado = 1/9 * kernelCuadrado
kernelDiagonal = np.array(([1] + [0]*5)*4 + [1]).reshape(5,5)
kernelDiagonal = 1/5 * kernelDiagonal

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
file_name = 'imagenes.zip'
download_file_from_google_drive('1EX0i7Tv8L3vECMfnjIo_cVFO4rhT1_qK', file_name)
with zipfile.ZipFile(file_name, 'r') as z:
    z.extractall()
 
#Obtener los paths de cada una de las imagenes
path = os.path
files = [f for f in glob.glob(os.path.join('some_images','*.png'), recursive=True)]

#Obtener un array con todas las imagenes y crear los arrays para 
images = [rgb2gray(io.imread(os.path.join(f))) for f in files]
images_squares = []
images_lines = []

#Aplicar la cross-correlacion a cada imagen de la base de datos suando los kernels de cuadrado y de linea
for i in range(0,5):
    image = images[i]
    plt.subplot(5,3,1+3*i)
    plt.title(f'Imagen {(i+1)}')
    plt.imshow(image,cmap='gray')
    
    image1 = MyCCorrelation_201424311_201617853(image, kernelCuadrado,'fill')
    images_squares.append(image1)
    plt.subplot(5,3,3*i+2)
    plt.title('Cuadrados')
    plt.imshow(image1,cmap='gray')
    
    image2 = MyCCorrelation_201424311_201617853(image, kernelDiagonal,'fill')
    images_lines.append(image2)
    plt.subplot(5,3,3*i+3)
    plt.title('Lineas')
    plt.imshow(image2,cmap='gray')
  
plt.tight_layout()
plt.show()

for i in range(0,5):
    image = images[i+5]
    plt.subplot(5,3,1+3*i)
    plt.title(f'Imagen {(i+6)}')
    plt.imshow(image,cmap='gray')
    
    image1 = MyCCorrelation_201424311_201617853(image, kernelCuadrado,'fill')
    images_squares.append(image1)
    plt.subplot(5,3,3*i+2)
    plt.title('Cuadrados')
    plt.imshow(image1,cmap='gray', vmin=0, vmax=1)
    
    image2 = MyCCorrelation_201424311_201617853(image, kernelDiagonal,'fill')
    images_lines.append(image2)
    plt.subplot(5,3,3*i+3)
    plt.title('Lineas')
    plt.imshow(image2,cmap='gray')

plt.tight_layout()    
plt.show()

#Obtener y visualizar los histogramas imagenes al ser filtrada con cada kernel
for i in range(0,5):
    image = images[i]
    plt.subplot(5,3,1+3*i)
    plt.title(f'Imagen {(i+1)}')
    plt.imshow(image,cmap='gray')
    
    image1 = images_squares[i]
    plt.subplot(5,3,3*i+2)
    plt.title('Cuadrados')
    plt.hist(image1.flatten())
    
    image2 = images_lines[i]
    plt.subplot(5,3,3*i+3)
    plt.title('Lineas')
    plt.hist(image2.flatten())

plt.tight_layout()    
plt.show()

for i in range(0,5):
    image = images[i+5]
    plt.subplot(5,3,1+3*i)
    plt.title(f'Imagen {(i+6)}')
    plt.imshow(image,cmap='gray')
    
    image1 = images_squares[i+5]
    plt.subplot(5,3,3*i+2)
    plt.title('Cuadrados')
    plt.hist(image1.flatten())
    
    image2 = images_lines[i+5]
    plt.subplot(5,3,3*i+3)
    plt.title('Lineas')
    plt.hist(image2.flatten())

plt.tight_layout()    
plt.show()

#Arreglo con las clases correctas para cada imagen
y_true = ['Lineas']*3 + ['Cuadrados']*3 + ['Lineas']*2 + ['Cuadrados']*2
y_pred = []
data = []

#Calculo de la clase para cada imagen segun el resultado de aplicar el kernel
for i in range(0,10):
    image1 = images_squares[i]
    image2 = images_lines[i]
    #Valor maximo de la respuesta de la imagen a cada filtro
    max_squares = np.max(image1)
    max_lines = np.max(image2)
    #Se asigna a la imagen la clase con el mayor valor
    if(max_squares<max_lines): clase = 'Lineas'
    else: clase = 'Cuadrados'
    y_pred.append(clase)
    data.append([i+1, max_squares, max_lines, clase])

#
df = pd.DataFrame(data, columns = ['Imagen', 'Cuadrados', 'Lineas', 'Clase']) 
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
    
aca = accuracy_score(y_true, y_pred)
print(f'El ACA obtenido es de {aca}')