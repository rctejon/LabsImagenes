#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io
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
    alpha+=0.01
    alpha = round(alpha, 2)
    plt.title(f'Alpha: {alpha}')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

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