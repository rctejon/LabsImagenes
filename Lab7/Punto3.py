#Librerias utilizadas
import numpy as np
import matplotlib.pyplot as plt
import os, requests
import cv2
from skimage import io
from skimage import morphology as morph
from skimage.color import rgb2gray

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

if not os.path.exists('ims'):
    os.mkdir('ims')
    
#Descarga de la imagen desde google drive
file_name = 'ims/bacillus.jpg'
download_file_from_google_drive('1cMCbrI9KtLEH8RztxyAXx2zmDoZFNg-K', file_name)          
image = io.imread(os.path.join('ims',"bacillus.jpg")).astype(np.uint8)
image = rgb2gray(image)/255

def geodesic_dilation(image, mask, structuring_element):
    img = image.copy()
    while True:
        temp = img
        img = cv2.dilate(img,structuring_element,iterations = 1)
        img = np.minimum(img, mask)
        if((temp == img).all()):
            return img

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,10))
erosion = cv2.dilate(image,kernel,iterations = 1)
geo = geodesic_dilation(erosion,image,kernel)
closure = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

plt.title(f'Closure')
plt.imshow(closure, cmap='gray')
plt.show()
