#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io
import scipy.io as scio
from skimage.color import rgb2gray

#
# Punto 7.1
#
print('Inicia punto 7.1')

#Descargar imagenes a color de internet
image_url = "https://i0.wp.com/ascienceenthusiast.com/wp-content/uploads/2019/02/Eminem001A.jpg?w=605&ssl=1"
r = requests.get(image_url)
with open("eminem1.jpg", "wb") as f:
    f.write(r.content)
image_url = "https://i1.wp.com/ascienceenthusiast.com/wp-content/uploads/2019/02/Eminem001B.jpg?w=605&ssl=1"
r = requests.get(image_url)
with open("eminem2.jpg", "wb") as f:
    f.write(r.content)

#Cargar las imagenes en la variables eminem1 y eminem2
eminem1 = io.imread(os.path.join("eminem1.jpg"))
eminem2 = io.imread(os.path.join("eminem2.jpg"))

# Calculo de suma de imagenes variando alpha 
alpha=0.
while alpha<=1.00:
    # Se realiza la suma de las imagenes y se convierte el tipo del arreglo a enteron de 8 bits sin signo
    imagen=np.add((1-alpha)*eminem1,alpha*eminem2).astype(np.uint8)
    plt.imshow(imagen)
    plt.title(f'Alpha: {alpha}')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()    
    alpha+=0.01
    alpha = round(alpha, 2)

input("Press Enter to continue...")

# Calculo promedio aritmetico
imagen=np.add(0.5*eminem1,0.5*eminem2).astype(np.uint8)
plt.imshow(imagen)
plt.title('Promedio aritmetico')
plt.show()
input("Press Enter to continue...")

# Calculo promedio aritmetico
imagen=np.sqrt(np.multiply(eminem1.astype(np.uint16),eminem2.astype(np.uint16))).astype(np.uint8)
plt.imshow(imagen)
plt.title('Promedio geometrico')
plt.show()
input("Press Enter to continue...")

#Descargar imagen en escala de gris
image_url = "https://onlineimagetools.com/images/examples-onlineimagetools/black-dog-gray-srgb.png"
r = requests.get(image_url)
with open("perrito.jpg", "wb") as f:
    f.write(r.content)
perro = io.imread(os.path.join("perrito.jpg"))
perro = rgb2gray(perro)

#Calculo de los parametros para la elipse 
a = perro.shape[1] / 2
b = perro.shape[0] / 2

#Grid con las coordenadas de los pixeles de la imagen
x, y = np.meshgrid(np.arange(0, perro.shape[1], 1), np.arange(0, perro.shape[0], 1))
z = ((x - a)**2 / a**2) + ((y - b)**2 / b**2)

#Mascara para que si algun pixel se encuentra sobre o fuera de la elipse, cambiar la intensidad a 0 (negro)
mask = z>=1
perro[mask] = 0
plt.imshow(perro, cmap = 'gray', vmin = 0, vmax = 1)
plt.show()
input("Press Enter to continue...")

#
# Punto 7.2
#
print('Inicia punto 7.2')

mat = scio.loadmat('detection_results.mat')

anotaciones = mat['annotations']
predicciones = mat['predictions']
confianzas = mat['scores']

conf=0.
puntosP=[]
puntosR=[]
while conf<=1:
    TP=0
    FP=0
    FN=0
    for i in range(0,len(predicciones)):
        anot = anotaciones[i]
        pred = predicciones[i]
        score = confianzas[i][0]
        maxX = max(anot[0],pred[0])
        maxY = max(anot[1],pred[1])
        minX = min(anot[0]+anot[2].astype(np.int32),pred[0]+pred[2].astype(np.int32))
        minY = min(anot[1]+anot[3].astype(np.int32),pred[1]+pred[3].astype(np.int32))
        area=0
        if maxX<minX and maxY<minY:
            area = (minX-maxX)*(minY-maxY)
        total_area = anot[2].astype(np.int32)*anot[3] + pred[2].astype(np.int32)*pred[3] - area
        if conf<area/total_area:
            TP+=1
        else:
            FP+=1
    FN = len(anotaciones) - TP
    prec = TP/(TP+FP)
    reca = TP/(TP+FN)
    puntosP.append(prec)
    puntosR.append(reca)
    conf+=0.1
plt.plot(puntosR,puntosP)
plt.title('Curva Precision-Cobertura')
plt.xlabel('Cobertura')
plt.ylabel('Precision')
plt.show()
input("Press Enter to continue...")

#
# Punto 7.3
#
print('Inicia punto 7.3')

def my_histogram_equalizator(image, show_plot=True):
    grayscale_image = image[:,:,0]
    L = 256
    MN = grayscale_image.shape[0] * grayscale_image.shape[1]
    histogram = plt.hist(grayscale_image.flatten(), bins=range(L))
    n = histogram[0]
    s = n.copy()
    
    for k in range(len(n)):
        sums = sum(n[0:k+1].astype(np.uint16))
        s[k] = np.uint8(round((L-1)*sums/MN + 0.01, 0))
        
    equalized_image = np.array([s[pixel] for pixel in grayscale_image])
    
    if(show_plot):
        i = plt.subplot(1, 2, 1)
        i.set_title("Imagen de bajo contraste")
        plt.imshow(grayscale_image, cmap='gray', vmin=0, vmax=L-1)
        i = plt.subplot(1, 2, 2)
        i.set_title("Imagen ecualizada")
        plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=L-1)
        plt.show()
        input("Press Enter to continue...")
        
image_url = "https://ak2.picdn.net/shutterstock/videos/15390892/thumb/1.jpg"
r = requests.get(image_url)
with open("prueba.jpg", "wb") as f:
    f.write(r.content)
test = io.imread(os.path.join("prueba.jpg"))

my_histogram_equalizator(test)    