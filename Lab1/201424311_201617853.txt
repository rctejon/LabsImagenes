#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io
from skimage.color import rgb2gray

#
# Punto 5.1
#
print('Inicia punto 5.1')

#Descargar imagen a color de internet
image_url = "https://www.thesprucepets.com/thmb/wNsACo_CfCKLw5zcgUOH9NCgqs8=/960x0/filters:no_upscale():max_bytes(150000):strip_icc()/fennec-fox-85120553-57ffe0d85f9b5805c2b03554.jpg"
r = requests.get(image_url)
with open("imagen.png", "wb") as f:
    f.write(r.content)
#Cargar la imagen en la variable image y mostrarla
image = io.imread(os.path.join("imagen.png"))
plt.imshow(image)
plt.show()

input("Press Enter to continue...")


#Visualizacion de cada uno de los canales de color en escala de grises
plt.figure(figsize = (8, 6))
i = plt.subplot(2, 2, 1)
i.set_title("Original Image")
plt.imshow(image)
for i in range(3):
    i1 = plt.subplot(2, 2, i+2)
    i1.set_title(f"Channel {i+1}")
    plt.imshow(image[:,:,i], cmap = 'gray')
#Guardar la imagen generada por el subplot
plt.savefig("plot.png")
plt.show()

input("Press Enter to continue...")


#Convertir la imagen en un vector y obtener el histograma
plt.hist(image.flatten())
plt.show()

input("Press Enter to continue...")


#Convertir la imagen a escala de grises y obtener el histograma
plt.hist(rgb2gray(image).flatten())
plt.show()

input("Press Enter to continue...")


#
# Punto 5.2
#
print('Inicia punto 5.2')
# Se descarga la imagen desde la url
r=requests.get('https://3pw8zx30ta4c3jegjv14ssuv-wpengine.netdna-ssl.com/wp-content/uploads/sites/2/2017/06/17-NEU-902-Bain-650x450.jpg')

# Se guarda la imagen en el archivo acv.jpg
with open('acv.jpg','wb') as f:
    f.write(r.content)

# Se lee la imagen del archivo acv.jpg
img=io.imread('acv.jpg')

# Se muestra el primer canal de la imagen
plt.imshow(img[:,:,0])
plt.show()
input("Press Enter to continue...")

# se crea y se muestra el histograma de la imagen con 20 bins
plt.hist(img.flatten(), bins=20)
plt.show()

input("Press Enter to continue...")


# Seleccionamos umbral maximo y minimo de intensidad de un pixel
minThreshold=150
maxThreshold=200

# Se crea la mascara y se muestra al lado de la imagen real
imgMask=img[::,::,0].copy()
mask1=imgMask<minThreshold
mask2=imgMask>maxThreshold
mask3=(imgMask>=minThreshold) & (imgMask<=maxThreshold)
imgMask[mask1]=0
imgMask[mask2]=0
imgMask[mask3]=1
i1=plt.subplot(121)
plt.imshow(img[:,:,0])
i2=plt.subplot(122)
i1.set_title('Original Image')
i2.set_title('Mask')
plt.imshow(imgMask)
plt.show()

input("Press Enter to continue...")


# Se muestra la imagen con la mascara aplicada en escala de grises
plt.imshow(img[:,:,0]*imgMask,cmap='gray')
plt.show()

input("Press Enter to continue...")

#
# Punto 5.3
#
print('Inicia punto 5.3')
#Obtener una lista de los archivos en la carpeta mixed_Slices con extension .dcm
files = glob.glob(os.path.join('mixed_slices', '*.dcm'))
#Obtener los metadatos de las tomografias
x = pydicom.dcmread(os.path.join('mixed_slices','0.dcm'))
rows, cols = x.Rows, x.Columns
n_cuts = (123, 110, 114)

#Crear las variables para guardar los grupos de tomografias segun el paciente
vol1 = np.zeros((n_cuts[0], rows, cols))
vol2 = np.zeros((n_cuts[1], rows, cols))
vol3 = np.zeros((n_cuts[2], rows, cols))

#Iterar los archivos y poner cada corte en el volumen y posicion adecuados
for file in files:
    cut = pydicom.dcmread(file)
    name = cut.PatientName
    if name == "CHAOS^CT_SET_8":
        vol1[cut.InstanceNumber-1] = cut.pixel_array
    elif name == "CHAOS^CT_SET_10":
        vol2[cut.InstanceNumber-1] = cut.pixel_array
    elif name == "CHAOS^CT_SET_26":
        vol3[cut.InstanceNumber-1] = cut.pixel_array

# Visualizacion de los cortes del eje Z del paciente 1
plt.ion()
plt.show()
i=0
for tom in vol1:
    plt.imshow(tom,cmap='gray')
    i+=1
    plt.title(f'Corte(eje Z): {i} Paciente: 1')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

input("Press Enter to continue...")


# Visualizacion de los cortes del eje Z del paciente 2
i=0
for tom in vol2:
    plt.imshow(tom, cmap='gray',vmin=0,vmax=2000)
    i+=1
    plt.title(f'Corte(eje Z): {i} Paciente: 2')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

input("Press Enter to continue...")

i=0
# Visualizacion de los cortes del eje Z del paciente 3
for tom in vol3:
    plt.imshow(tom,cmap='gray')
    i+=1
    plt.title(f'Corte(eje Z): {i} Paciente: 3')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

input("Press Enter to continue...")


# Rotacion de 270 grados a la derecha con respecto al eje Z de las imagenes para revisar el eje Y del paciente 1 
m1 = np.rot90(vol1, axes=(0,2))
m2 = np.rot90(m1, axes=(0,2))
vol4 = np.rot90(m2, axes=(0,2))

# Visualizacion de los cortes del eje Y del paciente 1
i=0
for tom in vol4:
    plt.imshow(tom,cmap='gray')
    i+=1
    plt.title(f'Corte(eje Y): {i} Paciente: 1')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

input("Press Enter to continue...")

# Rotacion de 270 grados a la derecha con respecto al eje Z de las imagenes para revisar el eje Y del paciente 2
m1 = np.rot90(vol2, axes=(0,2))
m2 = np.rot90(m1, axes=(0,2))
vol5 = np.rot90(m2, axes=(0,2))

# Visualizacion de los cortes del eje Y del paciente 1
i=0
for tom in vol5:
    plt.imshow(tom,cmap='gray')
    i+=1
    plt.title(f'Corte(eje Y): {i} Paciente: 2')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

input("Press Enter to continue...")

# Rotacion de 270 grados a la derecha con respecto al eje Z de las imagenes para revisar el eje Y del paciente 3 
m1 = np.rot90(vol3, axes=(0,2))
m2 = np.rot90(m1, axes=(0,2))
vol6 = np.rot90(m2, axes=(0,2))

# Visualizacion de los cortes del eje Y del paciente 3
i=0
for tom in vol6:
    plt.imshow(tom,cmap='gray')
    i+=1
    plt.title(f'Corte(eje Y): {i} Paciente: 3')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()