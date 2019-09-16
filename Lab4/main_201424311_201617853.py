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

#
# Punto 5.1
#
print('Inicia punto 5.1')

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

#Se descarga y convierte a rgb una imagen de internet
husky = 'https://img.milanuncios.com/fg/2659/27/265927918_1.jpg?VersionId=RmbFrmcRlrZ6TOb_fQ23oeGThoI_rfAs'
r = requests.get(husky)
with open("husky.jpg", "wb") as f:
    f.write(r.content)
husky = rgb2gray(io.imread(os.path.join("husky.jpg")))

#Se inicializan los kernels
kernelA = np.array([1,1,1,0,0,0,0,0,0]).reshape(3,3)
kernelB = np.array([1,0,-1,2,0,-2,1,0,-1]).reshape(3,3)
kernelC = (1/9)*np.array([1,1,1,1,1,1,1,1,1]).reshape(3,3)

#Se ejecutan las funciones de croscorrelacion con el kernel A y condicion valid
image1=MyCCorrelation_201424311_201617853(husky,kernelA,'valid')
image2 = sc.correlate2d(husky,kernelA,mode='valid',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel A - valid')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()
input("Press Enter to continue...")

#Se ejecutan las funciones de croscorrelacion con el kernel A y condicion wrap
image1=MyCCorrelation_201424311_201617853(husky,kernelA,'wrap')
image2 = sc.correlate2d(husky,kernelA,mode='same',boundary='wrap')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel A - wrap')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()
input("Press Enter to continue...")

#Se ejecutan las funciones de croscorrelacion con el kernel A y condicion fill
image1=MyCCorrelation_201424311_201617853(husky,kernelA,'fill')
image2 = sc.correlate2d(husky,kernelA,mode='same',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel A - fill')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()
input("Press Enter to continue...")

#Se ejecutan las funciones de croscorrelacion con el kernel B y condicion valid
image1=MyCCorrelation_201424311_201617853(husky,kernelB,'valid')
image2 = sc.correlate2d(husky,kernelB,mode='valid',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel B - valid')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()
input("Press Enter to continue...")

#Se ejecutan las funciones de croscorrelacion con el kernel B y condicion wrap
image1=MyCCorrelation_201424311_201617853(husky,kernelB,'wrap')
image2 = sc.correlate2d(husky,kernelB,mode='same',boundary='wrap')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel B - wrap')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()
input("Press Enter to continue...")

#Se ejecutan las funciones de croscorrelacion con el kernel B y condicion fill
image1=MyCCorrelation_201424311_201617853(husky,kernelB,'fill')
image2 = sc.correlate2d(husky,kernelB,mode='same',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel B - fill')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()
input("Press Enter to continue...")

#Se ejecutan las funciones de croscorrelacion con el kernel C y condicion valid
image1=MyCCorrelation_201424311_201617853(husky,kernelC,'valid')
image2 = sc.correlate2d(husky,kernelC,mode='valid',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel C - valid')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()
input("Press Enter to continue...")

#Se ejecutan las funciones de croscorrelacion con el kernel C y condicion wrap
image1=MyCCorrelation_201424311_201617853(husky,kernelC,'wrap')
image2 = sc.correlate2d(husky,kernelC,mode='same',boundary='wrap')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel C - wrap')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()
input("Press Enter to continue...")

#Se ejecutan las funciones de croscorrelacion con el kernel C y condicion fill
image1=MyCCorrelation_201424311_201617853(husky,kernelC,'fill')
image2 = sc.correlate2d(husky,kernelC,mode='same',boundary='fill')

#Se calcula el error cuadratico medio
MSE=((image1 - image2)**2)/(len(image1)*len(image1[0]))

print(np.sum(MSE))

plt.suptitle('Kernel C - fill')
plt.subplot(121)
plt.title('MyCCorrelation_201424311_201617853')
plt.imshow(image1,cmap='gray')
plt.subplot(122)
plt.title('correlate2d')
plt.imshow(image2,cmap='gray')
plt.show()
input("Press Enter to continue...")

#
# Punto 5.2
#
print('Inicia punto 5.2')

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
vmax = np.max(images[0])

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
    plt.imshow(image1,cmap='gray', vmax = vmax)
    
    image2 = MyCCorrelation_201424311_201617853(image, kernelDiagonal,'fill')
    images_lines.append(image2)
    plt.subplot(5,3,3*i+3)
    plt.title('Lineas')
    plt.imshow(image2,cmap='gray', vmax = vmax)
 
plt.tight_layout()
plt.show()
input("Press Enter to continue...")

for i in range(0,5):
    image = images[i+5]
    plt.subplot(5,3,1+3*i)
    plt.title(f'Imagen {(i+6)}')
    plt.imshow(image,cmap='gray')
    
    image1 = MyCCorrelation_201424311_201617853(image, kernelCuadrado,'fill')
    images_squares.append(image1)
    plt.subplot(5,3,3*i+2)
    plt.title('Cuadrados')
    plt.imshow(image1,cmap='gray', vmax = vmax)
    
    image2 = MyCCorrelation_201424311_201617853(image, kernelDiagonal,'fill')
    images_lines.append(image2)
    plt.subplot(5,3,3*i+3)
    plt.title('Lineas')
    plt.imshow(image2,cmap='gray', vmax = vmax)

plt.tight_layout()    
plt.show()
input("Press Enter to continue...")

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
input("Press Enter to continue...")

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
input("Press Enter to continue...")

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

#Mostrar datos de la tabla que se quiere construir para el informe
df = pd.DataFrame(data, columns = ['Imagen', 'Cuadrados', 'Lineas', 'Clase']) 
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

#Calculo del ACA  
aca = accuracy_score(y_true, y_pred)
print(f'El ACA obtenido es de {aca}')

#
# Punto 5.3
#
print('Inicia punto 5.3')

mat = scio.loadmat('challenge_results.mat')

def myACA_201424311_201617853(gt,pred, print_matrix=True):
    confMatrix = np.zeros((5,5))
    ACA = 0
    for i in range(len(pred)):
        g = gt[i]-1
        p = pred[i]-1
        confMatrix[g,p]+=1    
    for row in confMatrix:
        suma = np.sum(row)
        row *= 1/suma
    for i in range(len(confMatrix)):
        ACA+=confMatrix[i,i]/len(confMatrix)
    if print_matrix:
        print(confMatrix)
        print(ACA)
    return (confMatrix,ACA)


gt = mat['gt'][0].astype(np.int8)
method1 = mat['method1'][0].astype(np.int8)
method2 = mat['method2'][0].astype(np.int8)
method3 = mat['method3'][0].astype(np.int8)

myACA_201424311_201617853(gt,method1)
myACA_201424311_201617853(gt,method2)
myACA_201424311_201617853(gt,method3)