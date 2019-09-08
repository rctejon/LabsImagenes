#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pydicom
import requests
import skimage.io as io

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