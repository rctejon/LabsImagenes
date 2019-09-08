#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io

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
while alpha<=1:
    # Se realiza la suma de las imagenes y se convierte el tipo del arreglo a enteron de 8 bits sin signo
    imagen=np.add((1-alpha)*eminem1,alpha*eminem2).astype(np.uint8)
    plt.imshow(imagen)
    alpha+=0.01
    plt.title(f'Alpha: {alpha}')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

input("Press Enter to continue...")

# Calculo promedio aritmetico
imagen=np.add(0.5*eminem1,0.5*eminem2).astype(np.uint8)
plt.imshow(imagen)
plt.show()
input("Press Enter to continue...")
