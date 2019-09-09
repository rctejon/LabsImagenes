#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom, sys
import requests
import skimage.io as io
from skimage.color import rgb2gray

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